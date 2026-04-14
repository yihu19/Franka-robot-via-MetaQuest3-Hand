// FR3 Pose Command Client (non-VR protocol)
//
// Purpose:
//  1) Receive absolute end-effector pose commands over UDP:
//       "<x> <y> <z> <qx> <qy> <qz> <qw>"
//     (also accepts "pose <x> ... <qw>" for compatibility)
//  2) Execute commands on FR3 via IK + Ruckig velocity control.
//  3) Publish measurements over UDP JSON:
//       robot0_eef_pos, robot0_eef_quat, robot0_joint_pos, robot0_joint_vel,
//       robot0_joint_ext_torque, timestamp
//  4) If no new command arrives, keep tracking/holding the last target pose.
//
// Usage:
//   ./franka_pose_cmd_client <robot-hostname> [receiver-ip] [receiver-port] [cmd-port] [bidexhand]
//
// Defaults:
//   receiver-ip:   127.0.0.1
//   receiver-port: 9093
//   cmd-port:      8890
//   bidexhand:     false

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>
#include <ruckig/ruckig.hpp>

#include "examples_common.h"
#include "weighted_ik.h"

template <typename T>
class SeqLock {
 public:
  SeqLock() : seq_(0) {}
  explicit SeqLock(const T& init) : data_(init), seq_(0) {}

  void store(const T& value) {
    uint32_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release);
    data_ = value;
    seq_.store(s + 2, std::memory_order_release);
  }

  T load() const {
    T result;
    uint32_t s;
    do {
      s = seq_.load(std::memory_order_acquire);
      while (s & 1u) {
        s = seq_.load(std::memory_order_acquire);
      }
      result = data_;
    } while (seq_.load(std::memory_order_acquire) != s);
    return result;
  }

  bool try_load(T& out, int max_retries = 4) const {
    for (int i = 0; i < max_retries; ++i) {
      uint32_t s = seq_.load(std::memory_order_acquire);
      if (s & 1u) {
        continue;
      }
      out = data_;
      if (seq_.load(std::memory_order_acquire) == s) {
        return true;
      }
    }
    return false;
  }

 private:
  T data_{};
  alignas(64) std::atomic<uint32_t> seq_;
};

struct PoseCommand {
  std::array<double, 3> pos{0.0, 0.0, 0.0};
  std::array<double, 4> quat{0.0, 0.0, 0.0, 1.0};  // x,y,z,w
  double gripper_btn   = 0.0;   // 1.0 = close, 0.0 = open (toggle on rising edge)
  double gripper_speed = 0.1;
  double gripper_force = 20.0;
  double eps_inner     = 0.04;
  double eps_outer     = 0.04;
  bool valid = false;
  uint64_t seq = 0;
};

struct IKTarget {
  std::array<double, 7> joints{};
  bool valid = false;
  uint64_t seq = 0;
};

struct JointSnapshot {
  std::array<double, 7> q{};
};

struct LoggedState {
  std::array<double, 3> pos{};
  std::array<double, 4> quat{};
  std::array<double, 7> joint_pos{};
  std::array<double, 7> joint_vel{};
  std::array<double, 7> joint_ext_torque{};
  double timestamp = 0.0;
};

class RobotStateSender {
 public:
  RobotStateSender(const std::string& target_ip, int target_port, size_t capacity_pow2 = 16384)
      : target_ip_(target_ip), target_port_(target_port) {
    size_t cap = 1;
    while (cap < capacity_pow2) {
      cap <<= 1;
    }
    capacity_ = cap;
    mask_ = capacity_ - 1;
    buffer_.resize(capacity_);

    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
      std::cerr << "[StateSender] Failed to create UDP socket" << std::endl;
    }

    memset(&servaddr_, 0, sizeof(servaddr_));
    servaddr_.sin_family = AF_INET;
    servaddr_.sin_port = htons(target_port_);
    inet_pton(AF_INET, target_ip_.c_str(), &servaddr_.sin_addr);
  }

  ~RobotStateSender() {
    stop();
    if (sockfd_ >= 0) {
      close(sockfd_);
    }
  }

  void start() {
    running_.store(true);
    writer_thread_ = std::thread([this]() { writerLoop(); });
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }
    if (writer_thread_.joinable()) {
      writer_thread_.join();
    }
  }

  void enqueue(const LoggedState& s) {
    size_t tail = tail_.load(std::memory_order_relaxed);
    size_t next = (tail + 1) & mask_;
    size_t head = head_.load(std::memory_order_acquire);
    if (next == head) {
      // full -> drop oldest
      head_.store((head + 1) & mask_, std::memory_order_release);
      drops_.fetch_add(1, std::memory_order_relaxed);
    }
    buffer_[tail] = s;
    tail_.store(next, std::memory_order_release);
  }

 private:
  void writerLoop() {
    while (running_.load(std::memory_order_relaxed)) {
      size_t head = head_.load(std::memory_order_acquire);
      size_t tail = tail_.load(std::memory_order_acquire);
      if (head == tail) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      LoggedState s = buffer_[head];
      head_.store((head + 1) & mask_, std::memory_order_release);

      if (sockfd_ < 0) {
        continue;
      }

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(6);
      ss << "{";
      ss << "\"robot0_eef_pos\": [" << s.pos[0] << ", " << s.pos[1] << ", " << s.pos[2] << "], ";
      ss << "\"robot0_eef_quat\": [" << s.quat[0] << ", " << s.quat[1] << ", " << s.quat[2] << ", " << s.quat[3] << "], ";
      ss << "\"robot0_joint_pos\": [";
      for (int i = 0; i < 7; ++i) {
        ss << s.joint_pos[i];
        if (i < 6) {
          ss << ", ";
        }
      }
      ss << "], ";
      ss << "\"robot0_joint_vel\": [";
      for (int i = 0; i < 7; ++i) {
        ss << s.joint_vel[i];
        if (i < 6) {
          ss << ", ";
        }
      }
      ss << "], ";
      ss << "\"robot0_joint_ext_torque\": [";
      for (int i = 0; i < 7; ++i) {
        ss << s.joint_ext_torque[i];
        if (i < 6) {
          ss << ", ";
        }
      }
      ss << "], ";
      ss << "\"timestamp\": " << s.timestamp;
      ss << "}\n";

      const std::string msg = ss.str();
      sendto(sockfd_, msg.c_str(), msg.size(), 0, reinterpret_cast<const struct sockaddr*>(&servaddr_),
             sizeof(servaddr_));
    }
  }

  std::string target_ip_;
  int target_port_;
  int sockfd_ = -1;
  struct sockaddr_in servaddr_ {};

  std::vector<LoggedState> buffer_;
  size_t capacity_ = 0;
  size_t mask_ = 0;
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};
  std::atomic<bool> running_{false};
  std::thread writer_thread_;
  std::atomic<uint64_t> drops_{0};
};

class PoseCommandClient {
 public:
  PoseCommandClient(int cmd_port, bool bidexhand)
      : cmd_port_(cmd_port),
        q7_min_(bidexhand ? -0.2 : -2.89),
        q7_max_(bidexhand ? 1.9 : 2.89) {}

  ~PoseCommandClient() {
    running_.store(false);
    if (cmd_thread_.joinable()) {
      cmd_thread_.join();
    }
    ik_thread_running_.store(false);
    if (ik_thread_.joinable()) {
      ik_thread_.join();
    }
    gripper_thread_running_.store(false);
    gripper_cv_.notify_one();
    if (gripper_thread_.joinable()) {
      gripper_thread_.join();
    }
    if (sender_) {
      sender_->stop();
    }
    if (cmd_socket_ >= 0) {
      close(cmd_socket_);
    }
  }

  void run(const std::string& robot_ip, const std::string& receiver_ip, int receiver_port) {
    setupNetworking();

    franka::Robot robot(robot_ip);
    setDefaultBehavior(robot);
    robot.setCollisionBehavior(
        {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
        {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
        {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
        {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

    // Initialize state, IK neutral pose, and default "hold-current" pose command.
    franka::RobotState init_state = robot.readOnce();
    JointSnapshot js_init;
    std::array<double, 7> neutral_pose{};
    for (int i = 0; i < 7; ++i) {
      js_init.q[i] = init_state.q[i];
      neutral_pose[i] = init_state.q[i];
    }
    joint_state_buf_.store(js_init);

    {
      Eigen::Matrix4d T = Eigen::Matrix4d::Map(init_state.O_T_EE.data());
      Eigen::Quaterniond q(T.block<3, 3>(0, 0));
      PoseCommand cmd;
      cmd.pos = {T(0, 3), T(1, 3), T(2, 3)};
      cmd.quat = {q.x(), q.y(), q.z(), q.w()};
      cmd.valid = true;
      cmd.seq = ++command_seq_;
      pose_cmd_buf_.store(cmd);
    }

    std::array<double, 7> base_joint_weights = {{
        3.0,  // joint 0
        6.0,  // joint 1
        1.5,  // joint 2
        1.5,  // joint 3
        1.0,  // joint 4
        1.0,  // joint 5
        1.0   // joint 6
    }};

    ik_solver_ = std::make_unique<WeightedIKSolver>(
        neutral_pose, 1.0, 2.0, 2.0, base_joint_weights, false);

    trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
    trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;

    for (size_t i = 0; i < 7; ++i) {
      ruckig_input_.max_velocity[i] = MAX_JOINT_VELOCITY[i];
      ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
      ruckig_input_.max_jerk[i] = MAX_JOINT_JERK[i];
      ruckig_input_.target_velocity[i] = 0.0;
      ruckig_input_.target_acceleration[i] = 0.0;
    }

    // Gripper init
    try {
      gripper_ = std::make_unique<franka::Gripper>(robot_ip);
      std::cout << "[PoseClient] Gripper initialized." << std::endl;
      gripper_->homing();
      std::cout << "[PoseClient] Gripper homed." << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[PoseClient] Warning: gripper init failed: " << e.what() << std::endl;
    }

    // Gripper worker thread (blocking grasp/move off the RT loop)
    gripper_thread_running_.store(true);
    gripper_thread_ = std::thread([this]() {
      while (gripper_thread_running_) {
        GripperCmd cmd;
        {
          std::unique_lock<std::mutex> lk(gripper_mutex_);
          gripper_cv_.wait(lk, [this]() {
            return !gripper_thread_running_ || gripper_requested_;
          });
          if (!gripper_thread_running_) break;
          cmd = pending_gripper_cmd_;
          gripper_requested_ = false;
        }
        if (!gripper_) continue;
        try {
          if (cmd.close) {
            gripper_->grasp(0.0, cmd.speed, cmd.force, cmd.epsilon_inner, cmd.epsilon_outer);
          } else {
            gripper_->move(0.08, cmd.speed);
          }
        } catch (const std::exception& e) {
          std::cerr << "[PoseClient] Gripper cmd failed: " << e.what() << std::endl;
        }
      }
    });

    sender_ = std::make_unique<RobotStateSender>(receiver_ip, receiver_port);
    sender_->start();

    cmd_thread_ = std::thread(&PoseCommandClient::commandThread, this);
    ik_thread_running_.store(true);
    ik_thread_ = std::thread(&PoseCommandClient::ikThreadLoop, this);

    std::cout << "[PoseClient] Running. cmd_port=" << cmd_port_ << ", state -> " << receiver_ip << ":"
              << receiver_port << std::endl;

    robot.control([this](const franka::RobotState& rs, franka::Duration period) {
      return this->rtCallback(rs, period);
    });
  }

 private:
  static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1kHz
  static constexpr double Q7_SEARCH_RANGE = 0.5;
  static constexpr double Q7_OPT_TOL = 1e-6;
  static constexpr int Q7_MAX_IT = 100;
  static constexpr std::array<double, 7> MAX_JOINT_VELOCITY = {
      1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0};
  static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {
      4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0};
  static constexpr std::array<double, 7> MAX_JOINT_JERK = {
      8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};

  std::atomic<bool> running_{true};
  int cmd_socket_ = -1;
  int cmd_port_;

  double q7_min_;
  double q7_max_;

  std::atomic<uint64_t> command_seq_{0};
  SeqLock<PoseCommand> pose_cmd_buf_;
  SeqLock<JointSnapshot> joint_state_buf_;
  SeqLock<IKTarget> ik_target_buf_;

  std::thread cmd_thread_;
  std::thread ik_thread_;
  std::atomic<bool> ik_thread_running_{false};

  // Gripper
  std::unique_ptr<franka::Gripper> gripper_;
  std::mutex              gripper_mutex_;
  std::condition_variable gripper_cv_;
  bool gripper_requested_ = false;
  bool gripper_is_open_   = true;
  bool prev_gripper_btn_  = false;
  struct GripperCmd {
    bool   close         = false;
    double speed         = 0.1;
    double force         = 20.0;
    double epsilon_inner = 0.04;
    double epsilon_outer = 0.04;
  } pending_gripper_cmd_;
  std::thread      gripper_thread_;
  std::atomic<bool> gripper_thread_running_{false};

  std::unique_ptr<WeightedIKSolver> ik_solver_;
  std::unique_ptr<RobotStateSender> sender_;
  std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
  ruckig::InputParameter<7> ruckig_input_;
  ruckig::OutputParameter<7> ruckig_output_;
  bool ruckig_initialized_ = false;
  uint64_t last_applied_seq_ = 0;

  static bool parsePoseCommand(const char* msg, PoseCommand& out) {
    double x, y, z, qx, qy, qz, qw;
    double gripper_btn = 0.0, speed = 0.1, force = 20.0, eps_i = 0.04, eps_o = 0.04, width = 0.0;

    // Try 13-value format: x y z qx qy qz qw gripper_btn speed force eps_inner eps_outer width
    int parsed = std::sscanf(msg, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                             &x, &y, &z, &qx, &qy, &qz, &qw,
                             &gripper_btn, &speed, &force, &eps_i, &eps_o, &width);
    if (parsed < 7) {
      // Try "pose x y z ..." compatibility format
      parsed = std::sscanf(msg, "pose %lf %lf %lf %lf %lf %lf %lf", &x, &y, &z, &qx, &qy, &qz, &qw);
      if (parsed != 7) return false;
    }

    const double qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (qn < 1e-10) return false;

    out.pos          = {x, y, z};
    out.quat         = {qx / qn, qy / qn, qz / qn, qw / qn};
    out.gripper_btn  = gripper_btn;
    out.gripper_speed = speed;
    out.gripper_force = force;
    out.eps_inner    = eps_i;
    out.eps_outer    = eps_o;
    out.valid        = true;
    return true;
  }

  void setupNetworking() {
    cmd_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (cmd_socket_ < 0) {
      throw std::runtime_error("Failed to create command socket");
    }

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(cmd_port_);
    if (bind(cmd_socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
      throw std::runtime_error("Failed to bind command socket");
    }

    struct timeval tv {};
    tv.tv_sec = 0;
    tv.tv_usec = 100000;  // 100ms
    setsockopt(cmd_socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }

  void commandThread() {
    char buf[512];
    struct sockaddr_in client_addr {};
    socklen_t client_len = sizeof(client_addr);

    while (running_.load(std::memory_order_relaxed)) {
      ssize_t n = recvfrom(cmd_socket_, buf, sizeof(buf) - 1, 0,
                           reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
      if (n <= 0) {
        continue;
      }
      buf[n] = '\0';

      PoseCommand cmd;
      if (!parsePoseCommand(buf, cmd)) {
        std::cerr << "[PoseClient] Invalid command: " << buf << std::endl;
        continue;
      }

      cmd.seq = ++command_seq_;
      pose_cmd_buf_.store(cmd);
    }
  }

  void ikThreadLoop() {
    uint64_t last_processed_seq = 0;

    while (ik_thread_running_.load(std::memory_order_relaxed)) {
      PoseCommand pose = pose_cmd_buf_.load();
      if (!pose.valid || pose.seq == last_processed_seq) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      last_processed_seq = pose.seq;

      JointSnapshot js = joint_state_buf_.load();

      Eigen::Quaterniond q(pose.quat[3], pose.quat[0], pose.quat[1], pose.quat[2]);
      q.normalize();
      Eigen::Matrix3d rot = q.toRotationMatrix();

      std::array<double, 3> target_pos = {pose.pos[0], pose.pos[1], pose.pos[2]};
      std::array<double, 9> target_rot = {
          rot(0, 0), rot(0, 1), rot(0, 2), rot(1, 0), rot(1, 1), rot(1, 2), rot(2, 0), rot(2, 1), rot(2, 2)};

      const double current_q7 = js.q[6];
      const double q7_start = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
      const double q7_end = std::min(2.89, current_q7 + Q7_SEARCH_RANGE);

      WeightedIKResult result = ik_solver_->solve_q7_optimized(
          target_pos, target_rot, js.q, q7_start, q7_end, Q7_OPT_TOL, Q7_MAX_IT);

      if (!result.success) {
        std::cerr << "[PoseClient] IK failed for seq=" << pose.seq << " target=["
                  << target_pos[0] << ", " << target_pos[1] << ", " << target_pos[2] << "]" << std::endl;
        continue;
      }

      IKTarget ik;
      ik.valid = true;
      ik.seq = pose.seq;
      ik.joints = result.joint_angles;
      ik_target_buf_.store(ik);
    }
  }

  franka::JointVelocities rtCallback(const franka::RobotState& rs, franka::Duration /*period*/) {
    JointSnapshot js;
    for (int i = 0; i < 7; ++i) {
      js.q[i] = rs.q[i];
    }
    joint_state_buf_.store(js);

    if (!ruckig_initialized_) {
      for (int i = 0; i < 7; ++i) {
        ruckig_input_.current_position[i] = rs.q[i];
        ruckig_input_.current_velocity[i] = 0.0;
        ruckig_input_.current_acceleration[i] = 0.0;
        ruckig_input_.target_position[i] = rs.q[i];
        ruckig_input_.target_velocity[i] = 0.0;
        ruckig_input_.target_acceleration[i] = 0.0;
      }
      ruckig_initialized_ = true;
    } else {
      for (int i = 0; i < 7; ++i) {
        ruckig_input_.current_position[i] = rs.q[i];
        ruckig_input_.current_velocity[i] = ruckig_output_.new_velocity[i];
        ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i];
      }
    }

    // Gripper toggle on rising edge of gripper_btn
    {
      PoseCommand latest_cmd;
      if (pose_cmd_buf_.try_load(latest_cmd) && latest_cmd.valid) {
        bool btn = latest_cmd.gripper_btn > 0.5;
        if (btn && !prev_gripper_btn_) {
          std::lock_guard<std::mutex> lk(gripper_mutex_);
          pending_gripper_cmd_.close         = gripper_is_open_;
          pending_gripper_cmd_.speed         = latest_cmd.gripper_speed;
          pending_gripper_cmd_.force         = latest_cmd.gripper_force;
          pending_gripper_cmd_.epsilon_inner = latest_cmd.eps_inner;
          pending_gripper_cmd_.epsilon_outer = latest_cmd.eps_outer;
          gripper_requested_ = true;
          gripper_cv_.notify_one();
          gripper_is_open_ = !gripper_is_open_;
        }
        prev_gripper_btn_ = btn;
      }
    }

    IKTarget target;
    if (ik_target_buf_.try_load(target) && target.valid && target.seq != last_applied_seq_) {
      for (int i = 0; i < 7; ++i) {
        ruckig_input_.target_position[i] = target.joints[i];
      }
      ruckig_input_.target_position[6] = std::max(q7_min_, std::min(q7_max_, ruckig_input_.target_position[6]));
      last_applied_seq_ = target.seq;
    }

    const ruckig::Result rr = trajectory_generator_->update(ruckig_input_, ruckig_output_);

    std::array<double, 7> vel{};
    if (rr == ruckig::Result::Working || rr == ruckig::Result::Finished) {
      for (int i = 0; i < 7; ++i) {
        vel[i] = ruckig_output_.new_velocity[i];
      }
    } else {
      for (double& v : vel) {
        v = 0.0;
      }
    }

    if (sender_) {
      LoggedState snap;
      Eigen::Matrix4d T = Eigen::Matrix4d::Map(rs.O_T_EE.data());
      Eigen::Quaterniond q(T.block<3, 3>(0, 0));
      snap.pos = {T(0, 3), T(1, 3), T(2, 3)};
      snap.quat = {q.x(), q.y(), q.z(), q.w()};
      for (int i = 0; i < 7; ++i) {
        snap.joint_pos[i] = rs.q[i];
        snap.joint_vel[i] = rs.dq[i];
        snap.joint_ext_torque[i] = rs.tau_ext_hat_filtered[i];
      }
      snap.timestamp =
          std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
      sender_->enqueue(snap);
    }

    if (!running_.load(std::memory_order_relaxed)) {
      return franka::MotionFinished(franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    }
    return franka::JointVelocities(vel);
  }
};

int main(int argc, char** argv) {
  if (argc < 2 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " <robot-hostname> [receiver-ip] [receiver-port] [cmd-port] [bidexhand]\n"
              << "  receiver-ip:   default 127.0.0.1\n"
              << "  receiver-port: default 9093\n"
              << "  cmd-port:      default 8890\n"
              << "  bidexhand:     true/false (default false)\n";
    return -1;
  }

  const std::string robot_host = argv[1];
  const std::string receiver_ip = (argc >= 3) ? argv[2] : "127.0.0.1";
  const int receiver_port = (argc >= 4) ? std::stoi(argv[3]) : 9093;
  const int cmd_port = (argc >= 5) ? std::stoi(argv[4]) : 8890;
  bool bidexhand = false;
  if (argc >= 6) {
    const std::string arg = argv[5];
    bidexhand = (arg == "true" || arg == "1");
  }

  try {
    PoseCommandClient client(cmd_port, bidexhand);
    client.run(robot_host, receiver_ip, receiver_port);
  } catch (const franka::Exception& e) {
    std::cerr << "Franka exception: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "std::exception: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}

