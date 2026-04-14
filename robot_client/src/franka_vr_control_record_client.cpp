// VR-Based Cartesian Teleoperation
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

struct VRCommand
{
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    double button_pressed = 0.0; // gripper open/close command
    double gripper_speed = 0.1;  // m/s
    double gripper_force = 20.0; // N
    double epsilon_inner = 0.005; // m
    double epsilon_outer = 0.005; // m
    bool has_valid_data = false;
};


struct RobotState
{
    

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

// Single-producer single-consumer ring buffer logger that writes JSON lines to a file
class RobotStateLogger {
public:
    explicit RobotStateLogger(const std::string &filename, size_t capacity_pow2 = 16384)
        : filename_(filename)
    {
        // ensure capacity is power of two
        size_t cap = 1;
        while (cap < capacity_pow2) cap <<= 1;
        capacity_ = cap;
        mask_ = capacity_ - 1;
        buffer_.resize(capacity_);
        head_.store(0);
        tail_.store(0);
    }

    ~RobotStateLogger() {
        stop();
    }

    void start() {
        running_.store(true);
        out_.open(filename_, std::ios::out | std::ios::app);
        writer_thread_ = std::thread([this]() { this->writerLoop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (writer_thread_.joinable()) writer_thread_.join();
        if (out_.is_open()) out_.close();
    }

    // Non-blocking enqueue; if buffer is full we drop the oldest entry to make room (no blocking in control loop)
    void enqueue(const LoggedState &s) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = (tail + 1) & mask_;
        size_t head = head_.load(std::memory_order_acquire);
        if (next == head) {
            // Buffer full: advance head to drop oldest entry (avoid blocking producer)
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

            if (out_.is_open()) {
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
                out_ << ss.str();
            }
        }
    }

    std::string filename_;
    std::ofstream out_;
    std::vector<LoggedState> buffer_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<bool> running_{false};
    std::thread writer_thread_;
    std::atomic<uint64_t> drops_{0};
};


class VRController
{
private:
    std::atomic<bool> running_{true};
    VRCommand current_vr_command_;
    std::mutex command_mutex_;

    int server_socket_;
    const int PORT = 8888;

    // VR mapping parameters
    struct VRParams
    {
        double vr_smoothing = 0.05;       // Less for more responsive control

        // Deadzones to prevent drift from small sensor noise
        double position_deadzone = 0.001;   // 1mm
        double orientation_deadzone = 0.03; // ~1.7 degrees

        // Workspace limits to keep the robot in a safe area
        double max_position_offset = 0.75;   // 75cm from initial position
    } params_;

    // VR Target Pose
    Eigen::Vector3d vr_target_position_;
    Eigen::Quaterniond vr_target_orientation_;

    // VR filtering state
    Eigen::Vector3d filtered_vr_position_{0, 0, 0};
    Eigen::Quaterniond filtered_vr_orientation_{1, 0, 0, 0};

    // Initial poses used as a reference frame
    Eigen::Affine3d initial_robot_pose_;
    Eigen::Vector3d initial_vr_position_{0, 0, 0};
    Eigen::Quaterniond initial_vr_orientation_{1, 0, 0, 0};
    bool vr_initialized_ = false;

    // Joint space tracking
    std::array<double, 7> current_joint_angles_;
    std::array<double, 7> neutral_joint_pose_;
    std::unique_ptr<WeightedIKSolver> ik_solver_;
    
    // Q7 limits
    double Q7_MIN;
    double Q7_MAX;
    bool bidexhand_;
    static constexpr double Q7_SEARCH_RANGE = 0.5; // look for q7 angle candidates in +/- this value in the current joint range 
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6; // Tolerance for optimization
    static constexpr int Q7_MAX_ITERATIONS = 100; // Max iterations for optimization

    // Ruckig trajectory generator for smooth joint space motion
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7> ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;
    
    // Gradual activation to prevent sudden movements
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5; // Faster activation
    
    // Gripper control state
    bool prev_button_pressed_ = false;
    bool gripper_is_open_ = true;  // Start assuming gripper is open

    // Gripper worker thread and synchronization
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
    } pending_gripper_cmd_;
    std::thread gripper_thread_;
    std::atomic<bool> gripper_thread_running_{false};
    std::unique_ptr<RobotStateLogger> logger_;
    // gripper status polling
    std::atomic<double> latest_gripper_width_{-1.0};
    std::thread gripper_status_thread_;
    std::atomic<bool> gripper_status_thread_running_{false};
    
    // Franka joint limits for responsive teleoperation 
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY = {1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0};     // Increase for responsiveness
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0}; // Increase for snappier response
    static constexpr std::array<double, 7> MAX_JOINT_JERK = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};  // Higher jerk for snappier response
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

public:
    VRController(bool bidexhand = true)
        : Q7_MIN(bidexhand ? -0.2 : -2.89), Q7_MAX(bidexhand ? 1.9 : 2.89), bidexhand_(bidexhand)
    {
        setupNetworking();
    }

    ~VRController()
    {
        running_ = false;
        // Stop gripper worker if running
        gripper_thread_running_ = false;
        gripper_cv_.notify_one();
        if (gripper_thread_.joinable()) {
            gripper_thread_.join();
        }
        if (logger_) {
            logger_->stop();
        }
        // stop gripper status polling thread
        gripper_status_thread_running_ = false;
        if (gripper_status_thread_.joinable()) {
            gripper_status_thread_.join();
        }
        close(server_socket_);
    }

    void setupNetworking()
    {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0)
        {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(PORT);

        if (bind(server_socket_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            throw std::runtime_error("Failed to bind socket");
        }

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT << " for VR pose and gripper control data" << std::endl;
    }

    void networkThread()
    {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_)
        {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr *)&client_addr, &client_len);

            if (bytes_received > 0)
            {
                buffer[bytes_received] = '\0';

                VRCommand cmd;
                int parsed_count = sscanf(buffer, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                                          &cmd.pos_x, &cmd.pos_y, &cmd.pos_z,
                                          &cmd.quat_x, &cmd.quat_y, &cmd.quat_z, &cmd.quat_w,
                                          &cmd.button_pressed, &cmd.gripper_speed, &cmd.gripper_force, &cmd.epsilon_inner, &cmd.epsilon_outer);

                if (parsed_count == 12)
                {
                    cmd.has_valid_data = true;

                    std::lock_guard<std::mutex> lock(command_mutex_);
                    current_vr_command_ = cmd;

                    if (!vr_initialized_)
                    {
                        initial_vr_position_ = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                        initial_vr_orientation_ = Eigen::Quaterniond(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                        filtered_vr_position_ = initial_vr_position_;
                        filtered_vr_orientation_ = initial_vr_orientation_;

                        vr_initialized_ = true;
                        std::cout << "VR reference pose initialized!" << std::endl;
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // This function's only job is to calculate the desired target pose from VR data.
    void updateVRTargets(const VRCommand &cmd)
    {
        if (!cmd.has_valid_data || !vr_initialized_)
        {
            return;
        }

        // Current VR pose
        Eigen::Vector3d vr_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond vr_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        vr_quat.normalize();

        // Smooth incoming VR data to reduce jitter
        double alpha = 1.0 - params_.vr_smoothing;
        filtered_vr_position_ = params_.vr_smoothing * filtered_vr_position_ + alpha * vr_pos;
        filtered_vr_orientation_ = filtered_vr_orientation_.slerp(alpha, vr_quat);

        // Calculate deltas from the initial VR pose
        Eigen::Vector3d vr_pos_delta = filtered_vr_position_ - initial_vr_position_;
        Eigen::Quaterniond vr_quat_delta = filtered_vr_orientation_ * initial_vr_orientation_.inverse();

        // Apply deadzones to prevent drift
        if (vr_pos_delta.norm() < params_.position_deadzone)
        {
            vr_pos_delta.setZero();
        }
        double rotation_angle = 2.0 * acos(std::abs(vr_quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone)
        {
            vr_quat_delta.setIdentity();
        }

        // Apply workspace limits
        if (vr_pos_delta.norm() > params_.max_position_offset)
        {
            vr_pos_delta = vr_pos_delta.normalized() * params_.max_position_offset;
        }

        // The final calculation just updates the vr_target_
        vr_target_position_ = initial_robot_pose_.translation() + vr_pos_delta;
        vr_target_orientation_ = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        vr_target_orientation_.normalize();
    }

    // Helper function to clamp q7 within limits
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }
    
    // Convert Eigen types to arrays for geofik interface
    std::array<double, 3> eigenToArray3(const Eigen::Vector3d& vec) const {
        return {vec.x(), vec.y(), vec.z()};
    }
    
    std::array<double, 9> quaternionToRotationArray(const Eigen::Quaterniond& quat) const {
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        return {rot(0,0), rot(0,1), rot(0,2),
                rot(1,0), rot(1,1), rot(1,2), 
                rot(2,0), rot(2,1), rot(2,2)};
    }

public:
    void run(const std::string &robot_ip, const std::string &log_path = "robot_state_log.jsonl")
    {
        try
        {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);
            
            // Initialize gripper (worker thread will execute gripper commands)
            try {
                gripper_ = std::make_unique<franka::Gripper>(robot_ip);
                std::cout << "Gripper initialized" << std::endl;
                gripper_->homing();
                std::cout << "Gripper homed successfully" << std::endl;
                // Start gripper status polling thread (non-RT) to capture width without blocking control callback
                gripper_status_thread_running_ = true;
                gripper_status_thread_ = std::thread([this]() {
                    while (gripper_status_thread_running_) {
                        try {
                            auto state = gripper_->readOnce();
                            // franka::GripperState typically exposes `width`
                            latest_gripper_width_.store(state.width, std::memory_order_relaxed);
                        } catch (const std::exception &e) {
                            // ignore polling errors; keep last value
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                });
            } catch (const std::exception &e) {
                std::cerr << "Warning: could not initialize/home gripper: " << e.what() << std::endl;
                // proceed without gripper (worker thread will simply no-op)
            }

            // Start gripper worker thread
            gripper_thread_running_ = true;
            gripper_thread_ = std::thread([this]() {
                while (gripper_thread_running_) {
                    GripperCmd cmd;
                    {
                        std::unique_lock<std::mutex> lk(gripper_mutex_);
                        gripper_cv_.wait(lk, [this]() { return !gripper_thread_running_ || gripper_requested_; });
                        if (!gripper_thread_running_) break;
                        cmd = pending_gripper_cmd_;
                        gripper_requested_ = false;
                    }
                    try {
                        if (!gripper_) {
                            std::cerr << "Gripper not initialized, skipping gripper command" << std::endl;
                            continue;
                        }
                        if (cmd.close) {
                            gripper_->grasp(0.02, cmd.speed, cmd.force, cmd.epsilon_inner, cmd.epsilon_outer);
                        } else {
                            gripper_->move(0.08, cmd.speed);
                        }
                    } catch (const franka::Exception &e) {
                        std::cerr << "Gripper worker exception: " << e.what() << std::endl;
                    } catch (const std::exception &e) {
                        std::cerr << "Gripper worker std::exception: " << e.what() << std::endl;
                    }
                }
            });

            // Move to a suitable starting joint configuration
            std::array<double, 7> q_goal;
            if (bidexhand_) {
                q_goal = {{0.0, -0.812, -0.123, -2.0, 0.0, 2.8, 0.9}};  // BiDexHand pose
            } else {
                q_goal = {{0.0, -0.48, 0.0, -2.0, 0.0, 1.57, -0.85}};   // Full range pose
            }
            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This example will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // Collision behavior
            robot.setCollisionBehavior(
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});

            // Joint impedance for smooth motion (instead of Cartesian)
            robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

            // Initialize poses from the robot's current state
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));
            
            // Initialize joint angles
            for (int i = 0; i < 7; i++) {
                current_joint_angles_[i] = state.q[i];
                neutral_joint_pose_[i] = q_goal[i];  // Use the initial joint configuration as neutral
            }
            
            // Create IK solver with neutral pose and weights
            // Joint weights for base stabilization: higher weights for base joints (0,1)
            std::array<double, 7> base_joint_weights = {{
                3.0,  // Joint 0 (base rotation) - high penalty for stability
                6.0,  // Joint 1 (base shoulder) - high penalty for stability  
                1.5,  // Joint 2 (elbow) - normal penalty
                1.5,  // Joint 3 (forearm) - normal penalty
                1.0,  // Joint 4 (wrist) - normal penalty
                1.0,  // Joint 5 (wrist) - normal penalty
                1.0   // Joint 6 (hand) - normal penalty
            }};
            
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_,
                1.0,  // manipulability weight
                2.0,  // neutral distance weight  
                2.0,  // current distance weight
                base_joint_weights,  // per-joint weights for base stabilization
                false // verbose = false for production use
            );
            
            // Initialize Ruckig trajectory generator (but don't set initial state yet)
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            
            // Set up joint limits for safe teleoperation (but don't set positions yet)
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i] = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i] = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i] = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            
            std::cout << "Ruckig trajectory generator configured with 7 DOFs" << std::endl;

            // Initialize VR targets to the robot's starting pose
            vr_target_position_ = initial_robot_pose_.translation();
            vr_target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            // Start logger (writes to local file asynchronously to avoid blocking the control loop)
            logger_ = std::make_unique<RobotStateLogger>(log_path);
            logger_->start();

            std::thread network_thread(&VRController::networkThread, this);

            std::cout << "Waiting for VR data..." << std::endl;
            while (!vr_initialized_ && running_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (vr_initialized_)
            {
                std::cout << "VR initialized! Starting real-time control." << std::endl;
                this->runVRControl(robot);
            }

            running_ = false;
            if (network_thread.joinable())
                network_thread.join();
            if (logger_) {
                logger_->stop();
            }
            // Stop gripper worker thread
            gripper_thread_running_ = false;
            gripper_cv_.notify_one();
            if (gripper_thread_.joinable()) {
                gripper_thread_.join();
            }
        }
        catch (const franka::Exception &e)
        {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    void runVRControl(franka::Robot &robot)
    {
        auto vr_control_callback = [this](
                                       const franka::RobotState &robot_state,
                                       franka::Duration period) -> franka::JointVelocities
        {
            // Update VR targets from latest command (~50Hz)
            VRCommand cmd;
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                cmd = current_vr_command_;
            }
            updateVRTargets(cmd);
            
            // Handle gripper control: detect button press edge and signal worker thread
            bool button_pressed = cmd.button_pressed > 0.5;
            if (button_pressed && !prev_button_pressed_) {
                // Button was just pressed - prepare a gripper command
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close = gripper_is_open_; // if open -> close, else open
                    pending_gripper_cmd_.speed = cmd.gripper_speed;
                    pending_gripper_cmd_.force = cmd.gripper_force;
                    pending_gripper_cmd_.epsilon_inner = cmd.epsilon_inner;
                    pending_gripper_cmd_.epsilon_outer = cmd.epsilon_outer;
                    gripper_requested_ = true;
                }
                gripper_cv_.notify_one();
                gripper_is_open_ = !gripper_is_open_;
            }
            prev_button_pressed_ = button_pressed;

            // Initialize Ruckig with actual robot state on first call
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = 0.0; // Start with zero velocity command
                    ruckig_input_.current_acceleration[i] = 0.0; // Start with zero acceleration
                    ruckig_input_.target_position[i] = robot_state.q[i]; // Start with current position as target
                    ruckig_input_.target_velocity[i] = 0.0; // Start with zero target velocity
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
                std::cout << "Ruckig initialized for velocity control!" << std::endl;
                std::cout << "Starting with zero velocity commands to smoothly take over control" << std::endl;
            } else {
                // Update current joint state for Ruckig using previous Ruckig output for continuity
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = ruckig_output_.new_velocity[i]; // Use our own velocity command for continuity
                    ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i]; // Use Ruckig's acceleration
                }
            }
            
            // Calculate activation factor for gradual activation
            auto current_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(current_time - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);
            
            // Solve IK for VR target pose to get target joint angles
            std::array<double, 3> target_pos = eigenToArray3(vr_target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(vr_target_orientation_);
            
            // Calculate q7 search range around current value
            double current_q7 = current_joint_angles_[6];
            // Use full Franka Q7 range for IK solving, not bidexhand limits
            double q7_start = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end = std::min(2.89, current_q7 + Q7_SEARCH_RANGE);
            
            // Solve IK with weighted optimization
            WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
                target_pos, target_rot, current_joint_angles_,
                q7_start, q7_end, Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS
            );
            
            // Debug output for velocity control
            static int debug_counter = 0;
            debug_counter++;
            
            if (debug_counter % 100 == 0) {
                std::cout << "IK: " << (ik_result.success ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m") << " | Joints: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(2) << current_joint_angles_[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << std::endl;
            }
            
            // Set Ruckig targets based on IK solution and gradual activation
            if (ruckig_initialized_) {
                if (ik_result.success) {
                    // Gradually blend from current position to IK solution for target position
                    for (int i = 0; i < 7; i++) {
                        double current_pos = current_joint_angles_[i];
                        double ik_target_pos = ik_result.joint_angles[i];
                        ruckig_input_.target_position[i] = current_pos + activation_factor * (ik_target_pos - current_pos);
                        // Always target zero velocity for smooth stops
                        ruckig_input_.target_velocity[i] = 0.0;
                    }
                    // Enforce q7 limits
                    ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
                }
                // If IK fails, keep previous targets (don't change target_position/velocity)
            }
            
            // Always run Ruckig to generate smooth velocity commands
            ruckig::Result ruckig_result = trajectory_generator_->update(ruckig_input_, ruckig_output_);
            
            std::array<double, 7> target_joint_velocities;
            
            if (ruckig_result == ruckig::Result::Working || ruckig_result == ruckig::Result::Finished) {
                // Use Ruckig's smooth velocity output
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency fallback: zero velocity to stop smoothly
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = 0.0;
                }
                if (debug_counter % 100 == 0) {
                    std::cout << "Ruckig error, using zero velocity for safety" << std::endl;
                }
            }
            
            // Debug output for the first few commands
            // if (debug_counter <= 10 || debug_counter % 100 == 0) {
            //     std::cout << "Target vel: ";
            //     for (int i = 0; i < 7; i++) std::cout << std::fixed << std::setprecision(4) << target_joint_velocities[i] << " ";
            //     std::cout << " [activation: " << std::setprecision(3) << activation_factor << "]" << std::endl;
            // }

            if (!running_)
            {
                return franka::MotionFinished(franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
            }

            // Capture a non-blocking snapshot of robot state and enqueue for async logging
            if (logger_) {
                LoggedState snap;
                // end-effector pose from robot_state
                Eigen::Matrix4d T = Eigen::Matrix4d::Map(robot_state.O_T_EE.data());
                Eigen::Vector3d ee_pos = T.block<3,1>(0,3);
                Eigen::Matrix3d R = T.block<3,3>(0,0);
                Eigen::Quaterniond ee_quat(R);
                for (int i = 0; i < 3; ++i) snap.pos[i] = ee_pos[i];
                snap.quat[0] = ee_quat.x(); snap.quat[1] = ee_quat.y(); snap.quat[2] = ee_quat.z(); snap.quat[3] = ee_quat.w();

                // external wrench (force/torque) at the end effector (estimates provided by libfranka)
                // Use O_F_ext_hat_K if available
                snap.force = {0.0, 0.0, 0.0};
                snap.torque = {0.0, 0.0, 0.0};
                try {
                    snap.force[0] = robot_state.K_F_ext_hat_K[0];
                    snap.force[1] = robot_state.K_F_ext_hat_K[1];
                    snap.force[2] = robot_state.K_F_ext_hat_K[2];
                    snap.torque[0] = robot_state.K_F_ext_hat_K[3];
                    snap.torque[1] = robot_state.K_F_ext_hat_K[4];
                    snap.torque[2] = robot_state.K_F_ext_hat_K[5];
                } catch(...) {}

                for (int i = 0; i < 7; ++i) {
                    snap.joint_pos[i] = robot_state.q[i];
                    snap.joint_vel[i] = robot_state.dq[i];
                }
                // External joint torques (filtered) reported by libfranka
                for (int i = 0; i < 7; ++i) {
                    snap.joint_ext_torque[i] = robot_state.tau_ext_hat_filtered[i];
                }

                // Read latest gripper width from polling thread (atomic, non-blocking)
                snap.gripper_width = latest_gripper_width_.load(std::memory_order_relaxed);

                snap.timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

                logger_->enqueue(snap);
            }
            return franka::JointVelocities(target_joint_velocities);
        };

        try
        {
            robot.control(vr_control_callback);
        }
        catch (const franka::ControlException &e)
        {
            std::cerr << "VR control exception: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname> [log_path] [bidexhand]" << std::endl;
        std::cerr << "  log_path: optional path to JSONL log file (default: robot_state_log.jsonl)" << std::endl;
        std::cerr << "  bidexhand: true or 1 to enable BiDexHand limits (default: false)" << std::endl;
        return -1;
    }

    // Defaults
    std::string log_path = "robot_state_log.jsonl";
    bool bidexhand = false;

    // If provided, second argument is log_path
    if (argc >= 3) {
        log_path = argv[2];
    }

    // If provided, third argument is bidexhand flag
    if (argc == 4) {
        std::string bidexhand_arg = argv[3];
        bidexhand = (bidexhand_arg == "true" || bidexhand_arg == "1");
    }

    try
    {
        VRController controller(bidexhand);
        // Add a signal handler to gracefully shut down on Ctrl+C
        controller.run(argv[1], log_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}