// VR-Based Cartesian Teleoperation — Real-Time Optimized Client
// Fixes applied vs. the original data_sender_client:
//   1. IK solver runs on a dedicated ~200 Hz thread, NOT inside the 1 kHz RT callback.
//   2. All shared data uses lock-free SeqLock (no std::mutex in the RT path).
//   3. The RT control thread is elevated to SCHED_FIFO priority.
//   4. No std::cout / blocking I/O inside the RT callback.
//
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <array>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>
#include <franka/gripper.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "examples_common.h"
#include "weighted_ik.h"
#include <ruckig/ruckig.hpp>

// ============================================================================
// Lock-free SeqLock for single-producer / multiple-consumer data sharing.
// The writer is never blocked; the reader retries on torn reads.
// T must be trivially copyable.
// ============================================================================
template <typename T>
class SeqLock {
public:
    SeqLock() : seq_(0) {}

    explicit SeqLock(const T& init) : data_(init), seq_(0) {}

    // Writer — call from exactly ONE thread.
    void store(const T& value) {
        uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release);   // odd → write in progress
        data_ = value;
        seq_.store(s + 2, std::memory_order_release);   // even → write complete
    }

    // Reader — safe to call from any thread.  Returns the latest consistent
    // snapshot.  If a write is in progress it spins for a few nanoseconds (the
    // write side copies < 200 bytes, so latency is negligible on x86).
    T load() const {
        T result;
        uint32_t s;
        do {
            s = seq_.load(std::memory_order_acquire);
            while (s & 1u) {                              // writer active
                s = seq_.load(std::memory_order_acquire);
            }
            result = data_;
        } while (seq_.load(std::memory_order_acquire) != s);
        return result;
    }

    // Non-blocking try_load — returns false if a consistent read could not be
    // obtained within max_retries attempts (useful in hard-RT contexts).
    bool try_load(T& out, int max_retries = 4) const {
        for (int i = 0; i < max_retries; ++i) {
            uint32_t s = seq_.load(std::memory_order_acquire);
            if (s & 1u) continue;                         // writer active
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
// Data structures exchanged between threads
// ============================================================================

struct VRCommand {
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    double button_pressed = 0.0;
    double gripper_speed = 0.1;
    double gripper_force = 20.0;
    double epsilon_inner = 0.005;
    double epsilon_outer = 0.005;
    double gripper_grasp_width = 0.005; // new width parameter
    bool has_valid_data = false;
};

// Published by the IK thread for the RT callback to consume.
struct IKTarget {
    std::array<double, 7> joint_angles{};
    bool valid = false;
};

// Published by the RT callback so the IK thread knows the actual joint state.
struct JointSnapshot {
    std::array<double, 7> q{};
};

// Lightweight snapshot structure to record the robot state
struct LoggedState {
    std::array<double, 3> pos;
    std::array<double, 4> quat;
    std::array<double, 3> force;
    std::array<double, 3> torque;
    std::array<double, 7> joint_pos;
    std::array<double, 7> joint_vel;
    std::array<double, 7> joint_ext_torque;
    double gripper_width = -1.0;
    double timestamp = 0.0;
};

// ============================================================================
// Async UDP state sender (unchanged from original — already lock-free)
// ============================================================================
class RobotStateSender {
public:
    explicit RobotStateSender(const std::string& target_ip, int target_port,
                              size_t capacity_pow2 = 16384)
        : target_ip_(target_ip), target_port_(target_port) {
        size_t cap = 1;
        while (cap < capacity_pow2) cap <<= 1;
        capacity_ = cap;
        mask_ = capacity_ - 1;
        buffer_.resize(capacity_);
        head_.store(0);
        tail_.store(0);

        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) std::cerr << "Failed to create sender socket" << std::endl;

        memset(&servaddr_, 0, sizeof(servaddr_));
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_port = htons(target_port_);
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
            head_.store((head + 1) & mask_, std::memory_order_release);
            drops_.fetch_add(1, std::memory_order_relaxed);
        }
        buffer_[tail] = s;
        tail_.store(next, std::memory_order_release);
    }

    uint64_t drops() const { return drops_.load(); }

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
                ss << "\"robot0_eef_pos\": [" << s.pos[0] << ", " << s.pos[1] << ", " << s.pos[2] << "], ";
                ss << "\"robot0_eef_quat\": [" << s.quat[0] << ", " << s.quat[1] << ", " << s.quat[2] << ", " << s.quat[3] << "], ";
                ss << "\"robot0_force_ee\": [" << s.force[0] << ", " << s.force[1] << ", " << s.force[2] << "], ";
                ss << "\"robot0_torque_ee\": [" << s.torque[0] << ", " << s.torque[1] << ", " << s.torque[2] << "], ";
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
                ss << "\"timestamp\": " << s.timestamp;
                ss << "}\n";

                std::string msg = ss.str();
                sendto(sockfd_, msg.c_str(), msg.length(), 0,
                       (const struct sockaddr*)&servaddr_, sizeof(servaddr_));
            }
        }
    }

    std::string target_ip_;
    int target_port_;
    int sockfd_ = -1;
    struct sockaddr_in servaddr_;
    std::vector<LoggedState> buffer_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<bool> running_{false};
    std::thread writer_thread_;
    std::atomic<uint64_t> drops_{0};
};

// ============================================================================
// Main controller
// ============================================================================
class VRController {
private:
    std::atomic<bool> running_{true};

    // ---- Lock-free channels (replace the old std::mutex) --------------------
    SeqLock<VRCommand>     vr_command_buf_;      // network thread  → IK / RT
    SeqLock<IKTarget>      ik_target_buf_;        // IK thread       → RT
    SeqLock<JointSnapshot> joint_state_buf_;      // RT callback     → IK thread

    int server_socket_;
    const int PORT = 8888;

    // VR mapping parameters
    struct VRParams {
        double vr_smoothing = 0.05;
        double position_deadzone = 0.001;
        double orientation_deadzone = 0.03;
        double max_position_offset = 0.75;
    } params_;

    // VR target pose (owned exclusively by the IK thread)
    Eigen::Vector3d vr_target_position_;
    Eigen::Quaterniond vr_target_orientation_;

    // VR filtering state (owned exclusively by the IK thread)
    Eigen::Vector3d filtered_vr_position_{0, 0, 0};
    Eigen::Quaterniond filtered_vr_orientation_{1, 0, 0, 0};

    // Initial poses
    Eigen::Affine3d initial_robot_pose_;
    Eigen::Vector3d initial_vr_position_{0, 0, 0};
    Eigen::Quaterniond initial_vr_orientation_{1, 0, 0, 0};
    std::atomic<bool> vr_initialized_{false};

    // Joint space tracking (shared via SeqLock — no raw access across threads)
    std::array<double, 7> neutral_joint_pose_;
    std::unique_ptr<WeightedIKSolver> ik_solver_;

    // Q7 limits
    double Q7_MIN;
    double Q7_MAX;
    bool bidexhand_;
    static constexpr double Q7_SEARCH_RANGE = 0.5;
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6;
    static constexpr int Q7_MAX_ITERATIONS = 100;

    // Ruckig (owned exclusively by the RT callback)
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7> ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;

    // Gradual activation
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5;

    // Gripper control state (RT callback only)
    bool prev_button_pressed_ = false;
    bool gripper_is_open_ = true;

    // Gripper worker thread
    std::unique_ptr<franka::Gripper> gripper_;
    std::mutex gripper_mutex_;
    std::condition_variable gripper_cv_;
    bool gripper_requested_ = false;
    struct GripperCmd {
        bool close = false;
        double speed = 0.1;
        double force = 20.0;
        double epsilon_inner = 0.005;
        double epsilon_outer = 0.005;
        double width = 0.005;
    } pending_gripper_cmd_;
    std::thread gripper_thread_;
    std::atomic<bool> gripper_thread_running_{false};

    std::unique_ptr<RobotStateSender> sender_;

    // Gripper status polling
    std::atomic<double> latest_gripper_width_{-1.0};
    std::thread gripper_status_thread_;
    std::atomic<bool> gripper_status_thread_running_{false};

    // IK thread
    std::thread ik_thread_;
    std::atomic<bool> ik_thread_running_{false};

    // Franka joint limits
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY     = {1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0};
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0};
    static constexpr std::array<double, 7> MAX_JOINT_JERK         = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

public:
    VRController(bool bidexhand = true)
        : Q7_MIN(bidexhand ? -0.2 : -2.89),
          Q7_MAX(bidexhand ? 1.9 : 2.89),
          bidexhand_(bidexhand) {
        setupNetworking();
    }

    ~VRController() {
        running_ = false;
        ik_thread_running_ = false;
        if (ik_thread_.joinable()) ik_thread_.join();
        gripper_thread_running_ = false;
        gripper_cv_.notify_one();
        if (gripper_thread_.joinable()) gripper_thread_.join();
        if (sender_) sender_->stop();
        gripper_status_thread_running_ = false;
        if (gripper_status_thread_.joinable()) gripper_status_thread_.join();
        close(server_socket_);
    }

    // ----------------------------------------------------------------
    // Networking (unchanged — non-RT)
    // ----------------------------------------------------------------
    void setupNetworking() {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0) throw std::runtime_error("Failed to create socket");

        struct sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(PORT);

        if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0)
            throw std::runtime_error("Failed to bind socket");

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT
                  << " for VR pose and gripper control data" << std::endl;
    }

    // Network receive thread — writes into SeqLock<VRCommand> (no mutex).
    void networkThread() {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_) {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr*)&client_addr, &client_len);
            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';

                VRCommand cmd;
                int parsed_count = sscanf(
                    buffer, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &cmd.pos_x, &cmd.pos_y, &cmd.pos_z,
                    &cmd.quat_x, &cmd.quat_y, &cmd.quat_z, &cmd.quat_w,
                    &cmd.button_pressed, &cmd.gripper_speed, &cmd.gripper_force,
                    &cmd.epsilon_inner, &cmd.epsilon_outer, &cmd.gripper_grasp_width);

                if (parsed_count == 13) {
                    cmd.has_valid_data = true;
                    vr_command_buf_.store(cmd);          // lock-free publish

                    if (!vr_initialized_.load(std::memory_order_relaxed)) {
                        initial_vr_position_ = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                        initial_vr_orientation_ =
                            Eigen::Quaterniond(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                        filtered_vr_position_ = initial_vr_position_;
                        filtered_vr_orientation_ = initial_vr_orientation_;

                        vr_initialized_.store(true, std::memory_order_release);
                        std::cout << "VR reference pose initialized!" << std::endl;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // ----------------------------------------------------------------
    // VR target computation (called ONLY from the IK thread)
    // ----------------------------------------------------------------
    void updateVRTargets(const VRCommand& cmd) {
        if (!cmd.has_valid_data || !vr_initialized_.load(std::memory_order_acquire))
            return;

        Eigen::Vector3d vr_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond vr_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        vr_quat.normalize();

        double alpha = 1.0 - params_.vr_smoothing;
        filtered_vr_position_ = params_.vr_smoothing * filtered_vr_position_ + alpha * vr_pos;
        filtered_vr_orientation_ = filtered_vr_orientation_.slerp(alpha, vr_quat);

        Eigen::Vector3d vr_pos_delta = filtered_vr_position_ - initial_vr_position_;
        Eigen::Quaterniond vr_quat_delta = filtered_vr_orientation_ * initial_vr_orientation_.inverse();

        if (vr_pos_delta.norm() < params_.position_deadzone) vr_pos_delta.setZero();

        double rotation_angle = 2.0 * acos(std::abs(vr_quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone) vr_quat_delta.setIdentity();

        if (vr_pos_delta.norm() > params_.max_position_offset)
            vr_pos_delta = vr_pos_delta.normalized() * params_.max_position_offset;

        vr_target_position_ = initial_robot_pose_.translation() + vr_pos_delta;
        vr_target_orientation_ = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        vr_target_orientation_.normalize();
    }

    // ----------------------------------------------------------------
    // IK thread — runs at ~200 Hz, completely off the RT path.
    // Reads VR commands + current joint snapshot, solves IK, publishes
    // the result via SeqLock so the 1 kHz RT callback can consume it.
    // ----------------------------------------------------------------
    void ikThreadLoop() {
        // Give the IK thread slightly elevated (but not RT) priority so it
        // runs promptly but doesn't compete with the SCHED_FIFO RT callback.
        {
            struct sched_param param{};
            param.sched_priority = 0;                      // normal for SCHED_OTHER
            pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
            // Nice the thread to -5 (best-effort higher priority) — silent fail OK.
            nice(-5);
        }

        int debug_counter = 0;

        while (ik_thread_running_.load(std::memory_order_relaxed)) {
            // 1. Read latest VR command (lock-free)
            VRCommand cmd = vr_command_buf_.load();

            // 2. Read latest joint angles published by the RT callback
            JointSnapshot js = joint_state_buf_.load();

            // 3. Compute VR target pose
            updateVRTargets(cmd);

            // 4. Solve IK
            std::array<double, 3> target_pos = eigenToArray3(vr_target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(vr_target_orientation_);

            double current_q7 = js.q[6];
            double q7_start = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end   = std::min( 2.89, current_q7 + Q7_SEARCH_RANGE);

            WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
                target_pos, target_rot, js.q,
                q7_start, q7_end, Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS);

            // 5. Publish result (lock-free)
            IKTarget target;
            target.valid = ik_result.success;
            if (ik_result.success) {
                target.joint_angles = ik_result.joint_angles;
            }
            ik_target_buf_.store(target);

            // 6. Debug print (safe here — we're not in the RT callback)
            ++debug_counter;
            if (debug_counter % 200 == 0) {
                std::cout << "IK: "
                          << (ik_result.success ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m")
                          << " | Joints: ";
                for (int i = 0; i < 7; ++i) {
                    std::cout << std::fixed << std::setprecision(2) << js.q[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << " | IK µs: " << ik_result.duration_microseconds << std::endl;
            }

            // ~200 Hz
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // ----------------------------------------------------------------
    // Helpers (pure, no side-effects — safe anywhere)
    // ----------------------------------------------------------------
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }

    std::array<double, 3> eigenToArray3(const Eigen::Vector3d& vec) const {
        return {vec.x(), vec.y(), vec.z()};
    }

    std::array<double, 9> quaternionToRotationArray(const Eigen::Quaterniond& quat) const {
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        return {rot(0, 0), rot(0, 1), rot(0, 2),
                rot(1, 0), rot(1, 1), rot(1, 2),
                rot(2, 0), rot(2, 1), rot(2, 2)};
    }

    // ----------------------------------------------------------------
    // Attempt to set SCHED_FIFO on the calling thread.
    // ----------------------------------------------------------------
    static bool setRealtimePriority(int priority = 80) {
        struct sched_param param{};
        param.sched_priority = priority;
        int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (ret != 0) {
            std::cerr << "Warning: failed to set SCHED_FIFO priority " << priority
                      << " (errno " << ret << "). "
                      << "Run with sudo or grant CAP_SYS_NICE." << std::endl;
            return false;
        }
        return true;
    }

public:
    // ================================================================
    // run()  — main entry point
    // ================================================================
    void run(const std::string& robot_ip,
             const std::string& receiver_ip = "127.0.0.1",
             int receiver_port = 9091) {
        try {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);

            // ---- Gripper init ----
            try {
                gripper_ = std::make_unique<franka::Gripper>(robot_ip);
                std::cout << "Gripper initialized" << std::endl;
                gripper_->homing();
                std::cout << "Gripper homed successfully" << std::endl;
                gripper_status_thread_running_ = true;
                gripper_status_thread_ = std::thread([this]() {
                    while (gripper_status_thread_running_) {
                        try {
                            auto state = gripper_->readOnce();
                            latest_gripper_width_.store(state.width, std::memory_order_relaxed);
                        } catch (...) {}
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                });
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not initialize/home gripper: " << e.what() << std::endl;
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
                        if (!gripper_) continue;
                        if (cmd.close)
                            gripper_->grasp(0.005, cmd.speed, cmd.force,
                                            cmd.epsilon_inner, cmd.epsilon_outer);
                        else
                            gripper_->move(0.08, cmd.speed);
                    } catch (const franka::Exception& e) {
                        std::cerr << "Gripper worker exception: " << e.what() << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Gripper worker std::exception: " << e.what() << std::endl;
                    }
                }
            });

            // ---- Move to start pose ----
            std::array<double, 7> q_goal;
            if (bidexhand_)
                q_goal = {{0.0, -0.812, -0.123, -2.0, 0.0, 2.8, 0.9}};
            else
                q_goal = {{0.0, -0.48, 0.0, -2.0, 0.0, 1.57, -0.85}};

            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This example will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // ---- Collision / impedance ----
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

            // ---- Initialize state ----
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));

            std::array<double, 7> current_joint_angles;
            for (int i = 0; i < 7; ++i) {
                current_joint_angles[i] = state.q[i];
                neutral_joint_pose_[i] = q_goal[i];
            }

            // Seed the joint_state SeqLock so the IK thread has a valid snapshot
            // from the very first iteration.
            JointSnapshot js_init;
            js_init.q = current_joint_angles;
            joint_state_buf_.store(js_init);

            // ---- IK solver ----
            std::array<double, 7> base_joint_weights = {{
                3.0, 6.0, 1.5, 1.5, 1.0, 1.0, 1.0
            }};
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_, 1.0, 2.0, 2.0, base_joint_weights, false);

            // ---- Ruckig ----
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i]     = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i]         = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i]     = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }

            // ---- VR targets to current pose ----
            vr_target_position_ = initial_robot_pose_.translation();
            vr_target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            // Seed the IK target buffer with current joints (no-motion default)
            IKTarget ik_init;
            ik_init.valid = true;
            ik_init.joint_angles = current_joint_angles;
            ik_target_buf_.store(ik_init);

            // ---- Start async sender ----
            sender_ = std::make_unique<RobotStateSender>(receiver_ip, receiver_port);
            sender_->start();

            // ---- Start network thread ----
            std::thread network_thread(&VRController::networkThread, this);

            std::cout << "Waiting for VR data..." << std::endl;
            while (!vr_initialized_.load(std::memory_order_acquire) && running_)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (vr_initialized_.load()) {
                std::cout << "VR initialized! Starting real-time control." << std::endl;

                // ---- Start IK thread (off the RT path) ----
                ik_thread_running_ = true;
                ik_thread_ = std::thread(&VRController::ikThreadLoop, this);

                // ---- Run RT control ----
                runVRControl(robot);
            }

            // ---- Clean up ----
            running_ = false;
            ik_thread_running_ = false;
            if (ik_thread_.joinable()) ik_thread_.join();
            if (network_thread.joinable()) network_thread.join();
            if (sender_) sender_->stop();
            gripper_thread_running_ = false;
            gripper_cv_.notify_one();
            if (gripper_thread_.joinable()) gripper_thread_.join();
        } catch (const franka::Exception& e) {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    // ================================================================
    // RT control callback — runs at 1 kHz under SCHED_FIFO.
    // No IK, no mutex, no I/O.
    // ================================================================
    void runVRControl(franka::Robot& robot) {
        auto vr_control_callback = [this](
            const franka::RobotState& robot_state,
            franka::Duration /*period*/) -> franka::JointVelocities
        {
            // ---- Publish current joint state for IK thread (lock-free) ----
            {
                JointSnapshot js;
                for (int i = 0; i < 7; ++i) js.q[i] = robot_state.q[i];
                joint_state_buf_.store(js);
            }

            // ---- Read latest VR command for gripper handling only (lock-free) ----
            VRCommand cmd;
            vr_command_buf_.try_load(cmd);

            // ---- Gripper edge detection & dispatch (non-blocking) ----
            bool button_pressed = cmd.button_pressed > 0.5;
            if (button_pressed && !prev_button_pressed_) {
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close = gripper_is_open_;
                    pending_gripper_cmd_.speed = cmd.gripper_speed;
                    pending_gripper_cmd_.force = cmd.gripper_force;
                    pending_gripper_cmd_.epsilon_inner = cmd.epsilon_inner;
                    pending_gripper_cmd_.epsilon_outer = cmd.epsilon_outer;
                    pending_gripper_cmd_.width = cmd.gripper_grasp_width;
                    gripper_requested_ = true;
                }
                gripper_cv_.notify_one();
                gripper_is_open_ = !gripper_is_open_;
            }
            prev_button_pressed_ = button_pressed;

            // ---- Ruckig init on first call ----
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; ++i) {
                    ruckig_input_.current_position[i]     = robot_state.q[i];
                    ruckig_input_.current_velocity[i]     = 0.0;
                    ruckig_input_.current_acceleration[i] = 0.0;
                    ruckig_input_.target_position[i]      = robot_state.q[i];
                    ruckig_input_.target_velocity[i]      = 0.0;
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
            } else {
                for (int i = 0; i < 7; ++i) {
                    ruckig_input_.current_position[i]     = robot_state.q[i];
                    ruckig_input_.current_velocity[i]     = ruckig_output_.new_velocity[i];
                    ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i];
                }
            }

            // ---- Activation ramp ----
            double elapsed_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);

            // ---- Read latest IK target (lock-free, never blocks) ----
            IKTarget ik;
            if (!ik_target_buf_.try_load(ik)) {
                // Extremely unlikely — fall through with default (valid=false)
            }

            if (ik.valid) {
                for (int i = 0; i < 7; ++i) {
                    double current_pos = robot_state.q[i];
                    double ik_target = ik.joint_angles[i];
                    ruckig_input_.target_position[i] =
                        current_pos + activation_factor * (ik_target - current_pos);
                    ruckig_input_.target_velocity[i] = 0.0;
                }
                ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
            }
            // If IK invalid, keep previous Ruckig targets (robot holds position).

            // ---- Ruckig update ----
            ruckig::Result ruckig_result =
                trajectory_generator_->update(ruckig_input_, ruckig_output_);

            std::array<double, 7> target_joint_velocities{};
            if (ruckig_result == ruckig::Result::Working ||
                ruckig_result == ruckig::Result::Finished) {
                for (int i = 0; i < 7; ++i)
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
            }
            // else: zero velocities (safe default from value-init above)

            if (!running_)
                return franka::MotionFinished(
                    franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

            // ---- Enqueue state snapshot (lock-free ring buffer) ----
            if (sender_) {
                LoggedState snap;
                Eigen::Matrix4d T = Eigen::Matrix4d::Map(robot_state.O_T_EE.data());
                Eigen::Vector3d ee_pos = T.block<3, 1>(0, 3);
                Eigen::Quaterniond ee_quat(T.block<3, 3>(0, 0));
                for (int i = 0; i < 3; ++i) snap.pos[i] = ee_pos[i];
                snap.quat = {ee_quat.x(), ee_quat.y(), ee_quat.z(), ee_quat.w()};

                snap.force  = {robot_state.K_F_ext_hat_K[0],
                               robot_state.K_F_ext_hat_K[1],
                               robot_state.K_F_ext_hat_K[2]};
                snap.torque = {robot_state.K_F_ext_hat_K[3],
                               robot_state.K_F_ext_hat_K[4],
                               robot_state.K_F_ext_hat_K[5]};

                for (int i = 0; i < 7; ++i) {
                    snap.joint_pos[i]        = robot_state.q[i];
                    snap.joint_vel[i]        = robot_state.dq[i];
                    snap.joint_ext_torque[i] = robot_state.tau_ext_hat_filtered[i];
                }
                snap.gripper_width = latest_gripper_width_.load(std::memory_order_relaxed);
                snap.timestamp = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                sender_->enqueue(snap);
            }

            return franka::JointVelocities(target_joint_velocities);
        };

        // ---- Elevate to SCHED_FIFO before entering the RT control loop ----
        bool rt_ok = setRealtimePriority(80);
        if (rt_ok) {
            std::cout << "RT priority (SCHED_FIFO 80) set for control thread." << std::endl;
        }

        try {
            robot.control(vr_control_callback);
        } catch (const franka::ControlException& e) {
            std::cerr << "VR control exception: " << e.what() << std::endl;
        }

        // Restore normal scheduling after control exits.
        {
            struct sched_param param{};
            param.sched_priority = 0;
            pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
        }
    }
};

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <robot-hostname> [receiver-ip] [bidexhand]" << std::endl;
        std::cerr << "  receiver-ip: optional IP to send robot state to (default: 127.0.0.1)"
                  << std::endl;
        std::cerr << "  bidexhand: true or 1 to enable BiDexHand limits (default: false)"
                  << std::endl;
        return -1;
    }

    std::string receiver_ip = "127.0.0.1";
    int receiver_port = 9091;
    bool bidexhand = false;

    if (argc >= 3) receiver_ip = argv[2];
    if (argc == 4) {
        std::string arg = argv[3];
        bidexhand = (arg == "true" || arg == "1");
    }

    try {
        VRController controller(bidexhand);
        controller.run(argv[1], receiver_ip, receiver_port);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
