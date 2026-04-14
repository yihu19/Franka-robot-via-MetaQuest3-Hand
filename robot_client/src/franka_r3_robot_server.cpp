// FR3 Robot Control Server
// Receives joint/pose/gripper commands from Python client via UDP (port 8889)
// Streams robot state to Python client via UDP (port 9092)
//
// Usage: ./franka_r3_robot_server <robot-ip> [state-receiver-ip] [state-receiver-port]
//
// Command format (text, space-separated):
//   joint <q1> <q2> <q3> <q4> <q5> <q6> <q7>
//   pose  <x> <y> <z> <qx> <qy> <qz> <qw>
//   gripper_open  <speed>
//   gripper_close <width> <speed> <force> <eps_inner> <eps_outer>
//   stop
//
// Ack sent back to command sender: "done" or "error <msg>"
// State JSON is streamed to state-receiver-ip:state-receiver-port

#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <array>
#include <chrono>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>
#include <Eigen/Dense>

#include "examples_common.h"
#include "weighted_ik.h"
#include <ruckig/ruckig.hpp>

// ============================================================================
// SeqLock — lock-free single-producer / multi-consumer.  T must be trivially copyable.
// ============================================================================
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
            while (s & 1u) s = seq_.load(std::memory_order_acquire);
            result = data_;
        } while (seq_.load(std::memory_order_acquire) != s);
        return result;
    }

    bool try_load(T& out, int max_retries = 4) const {
        for (int i = 0; i < max_retries; ++i) {
            uint32_t s = seq_.load(std::memory_order_acquire);
            if (s & 1u) continue;
            out = data_;
            if (seq_.load(std::memory_order_acquire) == s) return true;
        }
        return false;
    }

private:
    T data_{};
    alignas(64) std::atomic<uint32_t> seq_;
};

// ============================================================================
// Data structures
// ============================================================================

struct ControlTarget {
    std::array<double, 7> joint_angles{};
    bool valid = false;
    uint64_t cmd_id = 0;
};

struct JointSnapshot {
    std::array<double, 7> q{};
};

struct PoseTarget {
    std::array<double, 3> pos{};
    std::array<double, 4> quat{0.0, 0.0, 0.0, 1.0};  // x, y, z, w
    bool valid = false;
    uint64_t cmd_id = 0;
};

struct LoggedState {
    std::array<double, 3>  pos{};
    std::array<double, 4>  quat{};
    std::array<double, 3>  force{};
    std::array<double, 3>  torque{};
    std::array<double, 7>  joint_pos{};
    std::array<double, 7>  joint_vel{};
    std::array<double, 7>  joint_ext_torque{};
    double gripper_width = -1.0;
    double timestamp = 0.0;
};

// ============================================================================
// Async UDP state sender (lock-free ring buffer)
// ============================================================================
class RobotStateSender {
public:
    RobotStateSender(const std::string& target_ip, int target_port,
                     size_t capacity_pow2 = 16384)
        : target_ip_(target_ip), target_port_(target_port)
    {
        size_t cap = 1;
        while (cap < capacity_pow2) cap <<= 1;
        capacity_ = cap;
        mask_     = cap - 1;
        buffer_.resize(cap);
        head_.store(0);
        tail_.store(0);

        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) std::cerr << "[StateSender] Failed to create socket\n";

        memset(&servaddr_, 0, sizeof(servaddr_));
        servaddr_.sin_family      = AF_INET;
        servaddr_.sin_port        = htons(target_port_);
        inet_pton(AF_INET, target_ip_.c_str(), &servaddr_.sin_addr);
    }

    ~RobotStateSender() {
        stop();
        if (sockfd_ >= 0) close(sockfd_);
    }

    void start() {
        running_.store(true);
        writer_thread_ = std::thread([this]() { writerLoop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (writer_thread_.joinable()) writer_thread_.join();
    }

    void enqueue(const LoggedState& s) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = (tail + 1) & mask_;
        size_t head = head_.load(std::memory_order_acquire);
        if (next == head) {
            // Ring full — drop oldest entry
            head_.store((head + 1) & mask_, std::memory_order_release);
            drops_.fetch_add(1, std::memory_order_relaxed);
        }
        buffer_[tail] = s;
        tail_.store(next, std::memory_order_release);
    }

private:
    void writerLoop() {
        while (running_.load()) {
            size_t head = head_.load(std::memory_order_acquire);
            size_t tail = tail_.load(std::memory_order_acquire);
            if (head == tail) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            LoggedState s = buffer_[head];
            head_.store((head + 1) & mask_, std::memory_order_release);

            if (sockfd_ >= 0) {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "{";
                ss << "\"robot0_eef_pos\": ["     << s.pos[0]  << ", " << s.pos[1]  << ", " << s.pos[2]  << "], ";
                ss << "\"robot0_eef_quat\": ["    << s.quat[0] << ", " << s.quat[1] << ", " << s.quat[2] << ", " << s.quat[3] << "], ";
                ss << "\"robot0_force_ee\": ["    << s.force[0]  << ", " << s.force[1]  << ", " << s.force[2]  << "], ";
                ss << "\"robot0_torque_ee\": ["   << s.torque[0] << ", " << s.torque[1] << ", " << s.torque[2] << "], ";
                ss << "\"robot0_joint_pos\": [";
                for (int i = 0; i < 7; ++i) { ss << s.joint_pos[i]; if (i < 6) ss << ", "; }
                ss << "], ";
                ss << "\"robot0_joint_vel\": [";
                for (int i = 0; i < 7; ++i) { ss << s.joint_vel[i]; if (i < 6) ss << ", "; }
                ss << "], ";
                ss << "\"robot0_joint_ext_torque\": [";
                for (int i = 0; i < 7; ++i) { ss << s.joint_ext_torque[i]; if (i < 6) ss << ", "; }
                ss << "], ";
                ss << "\"robot0_gripper_qpos\": " << s.gripper_width << ", ";
                ss << "\"timestamp\": "           << s.timestamp;
                ss << "}\n";

                std::string msg = ss.str();
                sendto(sockfd_, msg.c_str(), msg.size(), 0,
                       reinterpret_cast<const struct sockaddr*>(&servaddr_), sizeof(servaddr_));
            }
        }
    }

    std::string  target_ip_;
    int          target_port_;
    int          sockfd_ = -1;
    struct sockaddr_in servaddr_{};
    std::vector<LoggedState> buffer_;
    size_t       capacity_ = 0;
    size_t       mask_     = 0;
    std::atomic<size_t>   head_{0};
    std::atomic<size_t>   tail_{0};
    std::atomic<bool>     running_{false};
    std::thread           writer_thread_;
    std::atomic<uint64_t> drops_{0};
};

// ============================================================================
// FR3RobotServer
// ============================================================================
class FR3RobotServer {
    static constexpr int    CMD_PORT             = 8889;
    static constexpr double CONTROL_CYCLE_TIME   = 0.001;  // 1 kHz
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY     = {1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0};
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0};
    static constexpr std::array<double, 7> MAX_JOINT_JERK         = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};

    std::atomic<bool> running_{true};

    // Inter-thread shared data (all lock-free)
    SeqLock<ControlTarget> control_target_buf_;
    SeqLock<JointSnapshot> joint_state_buf_;
    SeqLock<PoseTarget>    pose_target_buf_;

    // Motion completion tracking (written by RT callback, read by command thread)
    std::atomic<uint64_t> latest_done_cmd_id_{0};
    std::atomic<uint64_t> current_cmd_id_{0};
    // Written by IK thread when IK fails so command thread is not stuck
    std::atomic<uint64_t> ik_failed_cmd_id_{0};

    // Command socket
    int cmd_socket_ = -1;

    // Gripper
    std::unique_ptr<franka::Gripper> gripper_;
    std::mutex              gripper_mutex_;
    std::condition_variable gripper_cv_;
    bool                    gripper_requested_ = false;
    struct GripperCmd {
        bool   close         = false;
        double width         = 0.025;
        double speed         = 0.1;
        double force         = 50.0;
        double epsilon_inner = 0.03;
        double epsilon_outer = 0.02;
    } pending_gripper_cmd_;
    std::thread           gripper_thread_;
    std::atomic<bool>     gripper_thread_running_{false};
    std::atomic<double>   latest_gripper_width_{-1.0};
    std::thread           gripper_status_thread_;
    std::atomic<bool>     gripper_status_thread_running_{false};
    std::atomic<bool>     gripper_done_{false};

    // IK
    std::unique_ptr<WeightedIKSolver> ik_solver_;
    std::thread  ik_thread_;
    std::atomic<bool> ik_thread_running_{false};

    // State sender
    std::unique_ptr<RobotStateSender> sender_;

    // Ruckig (owned exclusively by RT callback)
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7>  ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool     ruckig_initialized_  = false;
    uint64_t last_seen_cmd_id_    = 0;   // RT-only
    bool     motion_in_progress_  = false; // RT-only

    // ----------------------------------------------------------------
    void setupNetworking() {
        cmd_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (cmd_socket_ < 0) throw std::runtime_error("Failed to create command socket");

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(CMD_PORT);
        if (bind(cmd_socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("Failed to bind command socket");

        // 1-second receive timeout so the loop can check running_
        struct timeval tv{1, 0};
        setsockopt(cmd_socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        std::cout << "[Server] Command socket listening on port " << CMD_PORT << std::endl;
    }

    // ----------------------------------------------------------------
    void waitForMotionDone(uint64_t cmd_id, int poll_ms = 10, int timeout_s = 60) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        while (running_) {
            if (latest_done_cmd_id_.load(std::memory_order_acquire) >= cmd_id) return;
            if (ik_failed_cmd_id_.load(std::memory_order_acquire) >= cmd_id)   return;
            if (std::chrono::steady_clock::now() >= deadline) {
                std::cerr << "[Server] waitForMotionDone timed out after " << timeout_s
                          << "s for cmd_id=" << cmd_id << std::endl;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        }
    }

    // ----------------------------------------------------------------
    // Command receiver thread — runs sequentially (one command at a time).
    // ----------------------------------------------------------------
    void commandThread() {
        char buf[512];
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        std::cout << "[Server] Command thread started." << std::endl;

        while (running_) {
            ssize_t n = recvfrom(cmd_socket_, buf, sizeof(buf) - 1, 0,
                                 reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
            if (n <= 0) continue;   // timeout or error — just loop
            buf[n] = '\0';
            std::string cmd(buf);
            // Trim trailing newline/whitespace
            while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' '))
                cmd.pop_back();

            auto send_ack = [&](const std::string& msg) {
                sendto(cmd_socket_, msg.c_str(), msg.size(), 0,
                       reinterpret_cast<const struct sockaddr*>(&client_addr), client_len);
            };

            // ---- joint ----
            if (cmd.rfind("joint", 0) == 0) {
                std::array<double, 7> q{};
                int parsed = sscanf(cmd.c_str(), "joint %lf %lf %lf %lf %lf %lf %lf",
                                    &q[0], &q[1], &q[2], &q[3], &q[4], &q[5], &q[6]);
                if (parsed != 7) { send_ack("error invalid joint command"); continue; }

                uint64_t id = ++current_cmd_id_;
                ControlTarget ct;
                ct.joint_angles = q;
                ct.valid  = true;
                ct.cmd_id = id;
                control_target_buf_.store(ct);

                std::cout << "[cmd] joint id=" << id << " q=["
                          << q[0] << " " << q[1] << " " << q[2] << " "
                          << q[3] << " " << q[4] << " " << q[5] << " " << q[6] << "]\n";

                waitForMotionDone(id);
                send_ack("done");

            // ---- pose ----
            } else if (cmd.rfind("pose", 0) == 0) {
                double x, y, z, qx, qy, qz, qw;
                int parsed = sscanf(cmd.c_str(), "pose %lf %lf %lf %lf %lf %lf %lf",
                                    &x, &y, &z, &qx, &qy, &qz, &qw);
                if (parsed != 7) { send_ack("error invalid pose command"); continue; }

                uint64_t id = ++current_cmd_id_;
                PoseTarget pt;
                pt.pos  = {x, y, z};
                pt.quat = {qx, qy, qz, qw};
                pt.valid  = true;
                pt.cmd_id = id;
                pose_target_buf_.store(pt);

                std::cout << "[cmd] pose id=" << id
                          << " pos=[" << x << " " << y << " " << z << "]"
                          << " quat=[" << qx << " " << qy << " " << qz << " " << qw << "]\n";

                waitForMotionDone(id);
                if (ik_failed_cmd_id_.load() >= id)
                    send_ack("error ik_failed");
                else
                    send_ack("done");

            // ---- gripper_open ----
            } else if (cmd.rfind("gripper_open", 0) == 0) {
                double speed = 0.1;
                sscanf(cmd.c_str(), "gripper_open %lf", &speed);

                std::cout << "[cmd] gripper_open speed=" << speed << "\n";

                gripper_done_.store(false);
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close = false;
                    pending_gripper_cmd_.speed = speed;
                    gripper_requested_         = true;
                }
                gripper_cv_.notify_one();
                while (!gripper_done_.load() && running_)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                send_ack("done");

            // ---- gripper_close ----
            } else if (cmd.rfind("gripper_close", 0) == 0) {
                double w = 0.025, speed = 0.1, force = 50.0, eps_i = 0.03, eps_o = 0.02;
                sscanf(cmd.c_str(), "gripper_close %lf %lf %lf %lf %lf",
                       &w, &speed, &force, &eps_i, &eps_o);

                std::cout << "[cmd] gripper_close w=" << w << " speed=" << speed
                          << " force=" << force << "\n";

                gripper_done_.store(false);
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close         = true;
                    pending_gripper_cmd_.width         = w;
                    pending_gripper_cmd_.speed         = speed;
                    pending_gripper_cmd_.force         = force;
                    pending_gripper_cmd_.epsilon_inner = eps_i;
                    pending_gripper_cmd_.epsilon_outer = eps_o;
                    gripper_requested_                 = true;
                }
                gripper_cv_.notify_one();
                while (!gripper_done_.load() && running_)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                send_ack("done");

            // ---- stop ----
            } else if (cmd == "stop") {
                JointSnapshot js = joint_state_buf_.load();
                ControlTarget ct;
                ct.joint_angles = js.q;
                ct.valid  = true;
                ct.cmd_id = ++current_cmd_id_;
                control_target_buf_.store(ct);
                std::cout << "[cmd] stop — holding current position\n";
                send_ack("done");

            } else {
                std::cerr << "[cmd] Unknown command: " << cmd << "\n";
                send_ack("error unknown_command");
            }
        }
    }

    // ----------------------------------------------------------------
    // IK thread — runs at ~200 Hz, converts Cartesian targets to joint targets.
    // ----------------------------------------------------------------
    void ikThreadLoop() {
        uint64_t last_processed_id = 0;

        while (ik_thread_running_.load(std::memory_order_relaxed)) {
            PoseTarget pt = pose_target_buf_.load();

            if (pt.valid && pt.cmd_id != last_processed_id) {
                last_processed_id = pt.cmd_id;

                JointSnapshot js = joint_state_buf_.load();

                // Build rotation matrix from quaternion (x, y, z, w)
                Eigen::Quaterniond q(pt.quat[3], pt.quat[0], pt.quat[1], pt.quat[2]);
                q.normalize();
                Eigen::Matrix3d rot = q.toRotationMatrix();
                std::array<double, 9> target_rot = {
                    rot(0,0), rot(0,1), rot(0,2),
                    rot(1,0), rot(1,1), rot(1,2),
                    rot(2,0), rot(2,1), rot(2,2)
                };

                double current_q7   = js.q[6];
                double q7_start     = std::max(-2.89, current_q7 - 0.5);
                double q7_end       = std::min( 2.89, current_q7 + 0.5);

                WeightedIKResult result = ik_solver_->solve_q7_optimized(
                    {pt.pos[0], pt.pos[1], pt.pos[2]}, target_rot, js.q,
                    q7_start, q7_end, 1e-6, 100);

                if (result.success) {
                    ControlTarget ct;
                    ct.joint_angles = result.joint_angles;
                    ct.valid  = true;
                    ct.cmd_id = pt.cmd_id;
                    control_target_buf_.store(ct);
                    std::cout << "[IK] success for cmd_id=" << pt.cmd_id
                              << " µs=" << result.duration_microseconds << "\n";
                } else {
                    std::cerr << "[IK] Failed for cmd_id=" << pt.cmd_id
                              << " — robot will hold position\n";
                    ik_failed_cmd_id_.store(pt.cmd_id, std::memory_order_release);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // ----------------------------------------------------------------
    static bool setRealtimePriority(int priority = 80) {
        struct sched_param param{};
        param.sched_priority = priority;
        int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (ret != 0) {
            std::cerr << "[RT] Warning: failed to set SCHED_FIFO priority " << priority
                      << " (errno " << ret << ")\n";
            return false;
        }
        return true;
    }

    // ----------------------------------------------------------------
    // 1 kHz RT control callback
    // ----------------------------------------------------------------
    franka::JointVelocities rtCallback(const franka::RobotState& rs,
                                       franka::Duration /*period*/)
    {
        // Publish current joint state (lock-free)
        {
            JointSnapshot js;
            for (int i = 0; i < 7; ++i) js.q[i] = rs.q[i];
            joint_state_buf_.store(js);
        }

        // Read latest control target (lock-free, never blocks)
        ControlTarget ct;
        control_target_buf_.try_load(ct);

        // --- Ruckig initialisation on first call ---
        if (!ruckig_initialized_) {
            for (int i = 0; i < 7; ++i) {
                ruckig_input_.current_position[i]     = rs.q[i];
                ruckig_input_.current_velocity[i]     = 0.0;
                ruckig_input_.current_acceleration[i] = 0.0;
                ruckig_input_.target_position[i]      = rs.q[i];  // hold current
                ruckig_input_.target_velocity[i]      = 0.0;
                ruckig_input_.target_acceleration[i]  = 0.0;
            }
            // Do NOT copy ct.cmd_id here — any command already in the buffer
            // must be detected by the "New target?" check below, otherwise the
            // motion never starts and the done-ack is never sent.
            ruckig_initialized_ = true;
        } else {
            for (int i = 0; i < 7; ++i) {
                // Use Ruckig's own output as the next current state (self-consistent).
                // In velocity-control mode the robot follows Ruckig's velocity commands,
                // so the planned position is more consistent than rs.q which has
                // tracking error and noise that prevents Ruckig::Finished from firing.
                ruckig_input_.current_position[i]     = ruckig_output_.new_position[i];
                ruckig_input_.current_velocity[i]     = ruckig_output_.new_velocity[i];
                ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i];
            }
        }

        // --- New target? ---
        if (ct.valid && ct.cmd_id != last_seen_cmd_id_) {
            for (int i = 0; i < 7; ++i)
                ruckig_input_.target_position[i] = ct.joint_angles[i];
            last_seen_cmd_id_  = ct.cmd_id;
            motion_in_progress_ = true;
        }

        // --- Ruckig update ---
        ruckig::Result ruckig_result =
            trajectory_generator_->update(ruckig_input_, ruckig_output_);

        // --- Detect motion completion ---
        if (motion_in_progress_ && ruckig_result == ruckig::Result::Finished) {
            motion_in_progress_ = false;
            latest_done_cmd_id_.store(last_seen_cmd_id_, std::memory_order_release);
        } else if (motion_in_progress_ && ruckig_result == ruckig::Result::Error) {
            std::cerr << "[Ruckig] Error for cmd_id=" << last_seen_cmd_id_
                      << " — resetting target to current position." << std::endl;
            // Reset Ruckig target to current position so it stays stable
            for (int i = 0; i < 7; ++i) {
                ruckig_input_.target_position[i]    = ruckig_input_.current_position[i];
                ruckig_input_.target_velocity[i]    = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            motion_in_progress_ = false;
            ik_failed_cmd_id_.store(last_seen_cmd_id_, std::memory_order_release);
        }

        std::array<double, 7> velocities{};
        if (ruckig_result == ruckig::Result::Working ||
            ruckig_result == ruckig::Result::Finished) {
            for (int i = 0; i < 7; ++i)
                velocities[i] = ruckig_output_.new_velocity[i];
        }

        if (!running_)
            return franka::MotionFinished(
                franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

        // --- Enqueue state snapshot (lock-free) ---
        if (sender_) {
            LoggedState snap;
            Eigen::Matrix4d T = Eigen::Matrix4d::Map(rs.O_T_EE.data());
            Eigen::Vector3d  pos  = T.block<3,1>(0,3);
            Eigen::Quaterniond ee_q(T.block<3,3>(0,0));
            for (int i = 0; i < 3; ++i) snap.pos[i] = pos[i];
            snap.quat   = {ee_q.x(), ee_q.y(), ee_q.z(), ee_q.w()};
            snap.force  = {rs.K_F_ext_hat_K[0], rs.K_F_ext_hat_K[1], rs.K_F_ext_hat_K[2]};
            snap.torque = {rs.K_F_ext_hat_K[3], rs.K_F_ext_hat_K[4], rs.K_F_ext_hat_K[5]};
            for (int i = 0; i < 7; ++i) {
                snap.joint_pos[i]        = rs.q[i];
                snap.joint_vel[i]        = rs.dq[i];
                snap.joint_ext_torque[i] = rs.tau_ext_hat_filtered[i];
            }
            snap.gripper_width = latest_gripper_width_.load(std::memory_order_relaxed);
            snap.timestamp = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            sender_->enqueue(snap);
        }

        return franka::JointVelocities(velocities);
    }

public:
    void run(const std::string& robot_ip,
             const std::string& state_receiver_ip = "127.0.0.1",
             int                state_receiver_port = 9092)
    {
        setupNetworking();

        franka::Robot robot(robot_ip);
        setDefaultBehavior(robot);

        // ---- Gripper init ----
        try {
            gripper_ = std::make_unique<franka::Gripper>(robot_ip);
            std::cout << "[Server] Homing gripper...\n";
            gripper_->homing();
            gripper_->move(0.06, 0.1);
            std::cout << "[Server] Gripper ready.\n";

            gripper_status_thread_running_ = true;
            gripper_status_thread_ = std::thread([this]() {
                while (gripper_status_thread_running_) {
                    try {
                        auto st = gripper_->readOnce();
                        latest_gripper_width_.store(st.width, std::memory_order_relaxed);
                    } catch (...) {}
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            });
        } catch (const std::exception& e) {
            std::cerr << "[Server] Warning: gripper init failed: " << e.what() << "\n";
        }

        // ---- Gripper worker thread ----
        gripper_thread_running_ = true;
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
                try {
                    if (!gripper_) { gripper_done_ = true; continue; }
                    if (cmd.close)
                        gripper_->grasp(cmd.width, cmd.speed, cmd.force,
                                        cmd.epsilon_inner, cmd.epsilon_outer);
                    else
                        gripper_->move(0.06, cmd.speed);
                } catch (const franka::Exception& e) {
                    std::cerr << "[Gripper] Exception: " << e.what() << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "[Gripper] Exception: " << e.what() << "\n";
                }
                gripper_done_.store(true, std::memory_order_release);
            }
        });

        // ---- Read initial robot state ----
        franka::RobotState init_state = robot.readOnce();
        {
            JointSnapshot js;
            for (int i = 0; i < 7; ++i) js.q[i] = init_state.q[i];
            joint_state_buf_.store(js);

            ControlTarget ct;
            ct.joint_angles = js.q;
            ct.valid  = true;
            ct.cmd_id = 0;
            control_target_buf_.store(ct);
        }

        // ---- IK solver ----
        std::array<double, 7> neutral_pose;
        for (int i = 0; i < 7; ++i) neutral_pose[i] = init_state.q[i];
        std::array<double, 7> base_weights = {3.0, 6.0, 1.5, 1.5, 1.0, 1.0, 1.0};
        ik_solver_ = std::make_unique<WeightedIKSolver>(
            neutral_pose, 1.0, 2.0, 2.0, base_weights, false);

        // ---- IK thread ----
        ik_thread_running_ = true;
        ik_thread_ = std::thread(&FR3RobotServer::ikThreadLoop, this);

        // ---- Ruckig ----
        trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
        trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
        for (size_t i = 0; i < 7; ++i) {
            ruckig_input_.max_velocity[i]     = MAX_JOINT_VELOCITY[i];
            ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
            ruckig_input_.max_jerk[i]         = MAX_JOINT_JERK[i];
            ruckig_input_.target_velocity[i]      = 0.0;
            ruckig_input_.target_acceleration[i]  = 0.0;
        }

        // ---- Collision / impedance ----
        robot.setCollisionBehavior(
            {{100,100,80,80,80,80,60}}, {{100,100,80,80,80,80,60}},
            {{100,100,80,80,80,80,60}}, {{100,100,80,80,80,80,60}},
            {{80,80,80,80,80,80}},      {{80,80,80,80,80,80}},
            {{80,80,80,80,80,80}},      {{80,80,80,80,80,80}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

        // ---- State sender ----
        sender_ = std::make_unique<RobotStateSender>(state_receiver_ip, state_receiver_port);
        sender_->start();
        std::cout << "[Server] Streaming robot state → " << state_receiver_ip
                  << ":" << state_receiver_port << "\n";

        // ---- Command thread ----
        std::thread cmd_thread(&FR3RobotServer::commandThread, this);

        std::cout << "[Server] Ready. Waiting for commands...\n";

        // ---- RT control loop (blocks until running_ = false) ----
        setRealtimePriority(80);
        try {
            robot.control([this](const franka::RobotState& rs, franka::Duration d)
                          -> franka::JointVelocities { return rtCallback(rs, d); });
        } catch (const franka::ControlException& e) {
            std::cerr << "[Server] Control exception: " << e.what() << "\n";
        }

        // ---- Cleanup ----
        running_ = false;
        ik_thread_running_ = false;
        if (ik_thread_.joinable())  ik_thread_.join();
        gripper_thread_running_ = false;
        gripper_cv_.notify_one();
        if (gripper_thread_.joinable())       gripper_thread_.join();
        gripper_status_thread_running_ = false;
        if (gripper_status_thread_.joinable()) gripper_status_thread_.join();
        if (sender_) sender_->stop();
        if (cmd_thread.joinable()) cmd_thread.join();
        if (cmd_socket_ >= 0) close(cmd_socket_);
    }
};

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <robot-ip> [state-receiver-ip] [state-receiver-port]\n"
                  << "  state-receiver-ip:   default 127.0.0.1\n"
                  << "  state-receiver-port: default 9092\n";
        return -1;
    }

    std::string state_ip   = (argc >= 3) ? argv[2] : "127.0.0.1";
    int         state_port = (argc >= 4) ? std::stoi(argv[3]) : 9092;

    try {
        FR3RobotServer server;
        server.run(argv[1], state_ip, state_port);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
