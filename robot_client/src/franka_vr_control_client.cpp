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
#include <cstring>  // for memset

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>
#include <franka/gripper.h>

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

class VRController
{
private:
    std::atomic<bool> running_{true};
    VRCommand current_vr_command_;
    std::mutex command_mutex_;

    // -----------------------------------------------------------------------
    // Networking – VR command receiver (port 8888)
    // -----------------------------------------------------------------------
    int server_socket_;
    const int PORT = 8888;

    // -----------------------------------------------------------------------
    // Networking – Robot state broadcaster (port 9091)
    // -----------------------------------------------------------------------
    int state_socket_;
    struct sockaddr_in state_addr_;
    const int STATE_PORT = 9091;
    const char* STATE_IP  = "127.0.0.1";  // publish on loopback; change to "0.0.0.0" to broadcast on all interfaces

    // VR mapping parameters
    struct VRParams
    {
        double vr_smoothing = 0.05;         // Lower = more responsive

        // Deadzones to prevent drift from small sensor noise
        double position_deadzone = 0.001;   // 1 mm
        double orientation_deadzone = 0.03; // ~1.7 degrees

        // Workspace limits to keep the robot in a safe area
        double max_position_offset = 0.75;  // 75 cm from initial position
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
    static constexpr double Q7_SEARCH_RANGE           = 0.5;   // ±0.5 rad search around current q7
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6;
    static constexpr int    Q7_MAX_ITERATIONS          = 20;

    // Ruckig trajectory generator for smooth joint space motion
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7>  ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;

    // Gradual activation to prevent sudden movements on control start
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5;

    // Gripper control state
    bool prev_button_pressed_ = false;
    bool gripper_is_open_     = true;  // Start assuming gripper is open

    // Actual gripper width, updated by the gripper worker thread
    std::atomic<double> gripper_width_m_{0.08};  // default open width

    // Gripper worker thread and synchronization
    std::unique_ptr<franka::Gripper> gripper_;
    std::mutex              gripper_mutex_;
    std::condition_variable gripper_cv_;
    bool gripper_requested_ = false;
    struct GripperCmd {
        bool   close         = false;
        double speed         = 0.1;
        double force         = 20.0;
        double epsilon_inner = 0.005;
        double epsilon_outer = 0.005;
    } pending_gripper_cmd_;
    std::thread         gripper_thread_;
    std::atomic<bool>   gripper_thread_running_{false};

    // -----------------------------------------------------------------------
    // State broadcast – dedicated off-RT thread
    // Snapshot holds only the fields needed by broadcastRobotState so the
    // RT callback copies as little data as possible before releasing.
    // -----------------------------------------------------------------------
    struct StateSnapshot {
        std::array<double, 16> O_T_EE{};
        std::array<double, 7>  q{};
        std::array<double, 7>  dq{};
        std::array<double, 7>  tau_ext_hat_filtered{};
        std::array<double, 6>  O_F_ext_hat_K{};
        double gripper_width = 0.0;
    };
    std::mutex           broadcast_mutex_;
    StateSnapshot        broadcast_snapshot_;
    std::atomic<bool>    broadcast_snapshot_ready_{false};
    std::thread          broadcast_thread_;

    // -----------------------------------------------------------------------
    // IK thread – runs solve_q7_optimized off the 1 kHz RT callback.
    // The RT callback posts target + joint angles at ~200 Hz via try_lock;
    // the IK thread solves and writes the result back. The RT callback reads
    // the result with try_lock and falls back to the cached solution.
    // -----------------------------------------------------------------------
    struct IKInput {
        std::array<double, 3> target_pos{};
        std::array<double, 9> target_rot{};
        std::array<double, 7> current_joints{};
        double q7_start = 0.0;
        double q7_end   = 0.0;
    };
    std::mutex           ik_input_mutex_;
    IKInput              ik_input_;
    std::atomic<bool>    ik_input_ready_{false};

    std::mutex           ik_output_mutex_;
    WeightedIKResult     ik_output_;
    bool                 ik_output_valid_ = false;

    WeightedIKResult     cached_ik_result_;
    bool                 cached_ik_valid_ = false;

    std::thread          ik_thread_;
    std::atomic<bool>    ik_thread_running_{false};

    // Franka joint limits for responsive teleoperation
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY     = {0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0};
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION  = {1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0};
    static constexpr std::array<double, 7> MAX_JOINT_JERK          = {3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0};
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

public:
    VRController(bool bidexhand = true)
        : Q7_MIN(bidexhand ? -0.2 : -2.89),
          Q7_MAX(bidexhand ?  1.9 :  2.89),
          bidexhand_(bidexhand)
    {
        setupNetworking();
        setupStateBroadcast();
    }

    ~VRController()
    {
        running_ = false;

        // Stop gripper worker
        gripper_thread_running_ = false;
        gripper_cv_.notify_one();
        if (gripper_thread_.joinable()) {
            gripper_thread_.join();
        }

        close(server_socket_);
        close(state_socket_);
    }

    // -----------------------------------------------------------------------
    // Setup – VR command UDP receiver
    // -----------------------------------------------------------------------
    void setupNetworking()
    {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0) {
            throw std::runtime_error("Failed to create VR command socket");
        }

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family      = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port        = htons(PORT);

        if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind VR command socket");
        }

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT
                  << " for VR pose and gripper control data" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Setup – Robot state UDP broadcaster
    // -----------------------------------------------------------------------
    void setupStateBroadcast()
    {
        state_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (state_socket_ < 0) {
            throw std::runtime_error("Failed to create state broadcast socket");
        }

        memset(&state_addr_, 0, sizeof(state_addr_));
        state_addr_.sin_family = AF_INET;
        state_addr_.sin_port   = htons(STATE_PORT);
        if (inet_pton(AF_INET, STATE_IP, &state_addr_.sin_addr) <= 0) {
            throw std::runtime_error("Invalid STATE_IP address");
        }

        std::cout << "Robot state will be broadcast to "
                  << STATE_IP << ":" << STATE_PORT << std::endl;
    }

    // -----------------------------------------------------------------------
    // makeSnapshot – copy the fields needed for broadcasting from a live
    //                franka::RobotState into a plain StateSnapshot.
    //                Safe to call from anywhere (no syscalls, no allocs).
    // -----------------------------------------------------------------------
    StateSnapshot makeSnapshot(const franka::RobotState& rs) const
    {
        StateSnapshot snap;
        snap.O_T_EE               = rs.O_T_EE;
        snap.q                    = rs.q;
        snap.dq                   = rs.dq;
        snap.tau_ext_hat_filtered = rs.tau_ext_hat_filtered;
        snap.O_F_ext_hat_K        = rs.O_F_ext_hat_K;
        snap.gripper_width        = gripper_width_m_.load();
        return snap;
    }

    // -----------------------------------------------------------------------
    // broadcastRobotState – formats and sends one UDP state packet.
    //
    // Must NOT be called from the 1 kHz RT callback (sendto + snprintf).
    // Call only from the dedicated broadcast thread or the pre-control loop.
    //
    // JSON keys match the Python FrankaRobot class in franka_robot.py:
    //   robot0_joint_pos        ← snap.q
    //   robot0_joint_vel        ← snap.dq
    //   robot0_eef_pos          ← snap.O_T_EE translation
    //   robot0_eef_quat         ← snap.O_T_EE rotation → [qx,qy,qz,qw]
    //   robot0_gripper_qpos     ← snap.gripper_width
    //   robot0_joint_ext_torque ← snap.tau_ext_hat_filtered
    //   robot0_force_ee         ← snap.O_F_ext_hat_K[0:3]
    //   robot0_torque_ee        ← snap.O_F_ext_hat_K[3:6]
    // -----------------------------------------------------------------------
    void broadcastRobotState(const StateSnapshot& snap)
    {
        // --- EE position from column-major 4×4 transform O_T_EE ---
        // Layout: O_T_EE[0..3] = col-0, [4..7] = col-1, [8..11] = col-2, [12..15] = col-3
        double ex = snap.O_T_EE[12];
        double ey = snap.O_T_EE[13];
        double ez = snap.O_T_EE[14];

        // --- Rotation matrix → quaternion ---
        // Column-major: R(row,col) = O_T_EE[col*4 + row]
        Eigen::Matrix3d R;
        R << snap.O_T_EE[0], snap.O_T_EE[4], snap.O_T_EE[8],
             snap.O_T_EE[1], snap.O_T_EE[5], snap.O_T_EE[9],
             snap.O_T_EE[2], snap.O_T_EE[6], snap.O_T_EE[10];
        Eigen::Quaterniond eq(R);
        eq.normalize();

        char buf[2048];
        int n = snprintf(buf, sizeof(buf),
            "{"
            "\"robot0_joint_pos\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_joint_vel\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_eef_pos\":[%.6f,%.6f,%.6f],"
            "\"robot0_eef_quat\":[%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_gripper_qpos\":%.6f,"
            "\"robot0_joint_ext_torque\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_force_ee\":[%.6f,%.6f,%.6f],"
            "\"robot0_torque_ee\":[%.6f,%.6f,%.6f]"
            "}\n",
            // joint positions
            snap.q[0], snap.q[1], snap.q[2], snap.q[3], snap.q[4], snap.q[5], snap.q[6],
            // joint velocities
            snap.dq[0], snap.dq[1], snap.dq[2], snap.dq[3], snap.dq[4], snap.dq[5], snap.dq[6],
            // EE position
            ex, ey, ez,
            // EE quaternion in scipy convention [qx, qy, qz, qw]
            eq.x(), eq.y(), eq.z(), eq.w(),
            // gripper width [m]
            snap.gripper_width,
            // external joint torques (tau_ext_hat_filtered)
            snap.tau_ext_hat_filtered[0], snap.tau_ext_hat_filtered[1],
            snap.tau_ext_hat_filtered[2], snap.tau_ext_hat_filtered[3],
            snap.tau_ext_hat_filtered[4], snap.tau_ext_hat_filtered[5],
            snap.tau_ext_hat_filtered[6],
            // EE external force [fx, fy, fz]
            snap.O_F_ext_hat_K[0], snap.O_F_ext_hat_K[1], snap.O_F_ext_hat_K[2],
            // EE external torque [tx, ty, tz]
            snap.O_F_ext_hat_K[3], snap.O_F_ext_hat_K[4], snap.O_F_ext_hat_K[5]
        );

        sendto(state_socket_, buf, n, 0,
               (struct sockaddr*)&state_addr_, sizeof(state_addr_));
    }

    // -----------------------------------------------------------------------
    // broadcastThreadFunc – off-RT thread that drains the snapshot buffer
    //                       and calls broadcastRobotState at ~100 Hz.
    // -----------------------------------------------------------------------
    void broadcastThreadFunc()
    {
        while (running_) {
            if (broadcast_snapshot_ready_.load(std::memory_order_acquire)) {
                StateSnapshot snap;
                {
                    std::lock_guard<std::mutex> lk(broadcast_mutex_);
                    snap = broadcast_snapshot_;
                    broadcast_snapshot_ready_.store(false, std::memory_order_relaxed);
                }
                broadcastRobotState(snap);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(9));  // ~110 Hz poll
        }
    }

    // -----------------------------------------------------------------------
    // ikThreadFunc – off-RT IK solver loop.
    //   Polls ik_input_ready_ at ~2 kHz (500 µs sleep).
    //   When new input is available, solves IK and writes to ik_output_.
    //   No syscalls are made from the RT callback to wake this thread.
    // -----------------------------------------------------------------------
    void ikThreadFunc()
    {
        while (ik_thread_running_) {
            if (ik_input_ready_.load(std::memory_order_acquire)) {
                IKInput inp;
                {
                    std::lock_guard<std::mutex> lk(ik_input_mutex_);
                    inp = ik_input_;
                    ik_input_ready_.store(false, std::memory_order_relaxed);
                }
                WeightedIKResult result = ik_solver_->solve_q7_optimized(
                    inp.target_pos, inp.target_rot, inp.current_joints,
                    inp.q7_start, inp.q7_end,
                    Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS);
                {
                    std::lock_guard<std::mutex> lk(ik_output_mutex_);
                    ik_output_       = result;
                    ik_output_valid_ = result.success;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    // -----------------------------------------------------------------------
    // networkThread – receives VR pose commands at ~1 kHz (non-blocking socket)
    // -----------------------------------------------------------------------
    void networkThread()
    {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_)
        {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr*)&client_addr, &client_len);

            if (bytes_received > 0)
            {
                buffer[bytes_received] = '\0';

                VRCommand cmd;
                int parsed_count = sscanf(buffer,
                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &cmd.pos_x,        &cmd.pos_y,          &cmd.pos_z,
                    &cmd.quat_x,       &cmd.quat_y,         &cmd.quat_z,      &cmd.quat_w,
                    &cmd.button_pressed, &cmd.gripper_speed, &cmd.gripper_force,
                    &cmd.epsilon_inner, &cmd.epsilon_outer);

                if (parsed_count == 12)
                {
                    cmd.has_valid_data = true;

                    std::lock_guard<std::mutex> lock(command_mutex_);
                    current_vr_command_ = cmd;

                    if (!vr_initialized_)
                    {
                        initial_vr_position_    = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                        initial_vr_orientation_ = Eigen::Quaterniond(
                            cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                        filtered_vr_position_    = initial_vr_position_;
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
    // -----------------------------------------------------------------------
    // updateVRTargets – maps filtered VR delta pose onto robot target pose
    // -----------------------------------------------------------------------
    void updateVRTargets(const VRCommand& cmd)
    {
        if (!cmd.has_valid_data || !vr_initialized_) {
            return;
        }

        Eigen::Vector3d   vr_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond vr_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        vr_quat.normalize();

        // Exponential smoothing
        double alpha = 1.0 - params_.vr_smoothing;
        filtered_vr_position_    = params_.vr_smoothing * filtered_vr_position_ + alpha * vr_pos;
        filtered_vr_orientation_ = filtered_vr_orientation_.slerp(alpha, vr_quat);

        // Delta from initial VR pose
        Eigen::Vector3d    vr_pos_delta  = filtered_vr_position_ - initial_vr_position_;
        Eigen::Quaterniond vr_quat_delta = filtered_vr_orientation_ * initial_vr_orientation_.inverse();

        // Position deadzone
        if (vr_pos_delta.norm() < params_.position_deadzone) {
            vr_pos_delta.setZero();
        }
        // Orientation deadzone
        double rotation_angle = 2.0 * acos(std::abs(vr_quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone) {
            vr_quat_delta.setIdentity();
        }

        // Workspace limit
        if (vr_pos_delta.norm() > params_.max_position_offset) {
            vr_pos_delta = vr_pos_delta.normalized() * params_.max_position_offset;
        }

        vr_target_position_    = initial_robot_pose_.translation() + vr_pos_delta;
        vr_target_orientation_ = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        vr_target_orientation_.normalize();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }

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
    // -----------------------------------------------------------------------
    // run – top-level entry point
    // -----------------------------------------------------------------------
    void run(const std::string& robot_ip)
    {
        try
        {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);

            // --- Gripper init ---
            try {
                gripper_ = std::make_unique<franka::Gripper>(robot_ip);
                std::cout << "Gripper initialized" << std::endl;
                gripper_->homing();
                std::cout << "Gripper homed successfully" << std::endl;
                // Read actual initial width
                franka::GripperState gs = gripper_->readOnce();
                gripper_width_m_.store(gs.width);
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not initialize/home gripper: " << e.what() << std::endl;
            }

            // --- Gripper worker thread ---
            // Executes blocking grasp/move commands off the RT loop.
            // After each command it reads back the actual gripper width.
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
                        if (!gripper_) {
                            std::cerr << "Gripper not initialized, skipping command" << std::endl;
                            continue;
                        }
                        if (cmd.close) {
                            gripper_->grasp(0.02, cmd.speed, cmd.force,
                                            cmd.epsilon_inner, cmd.epsilon_outer);
                        } else {
                            gripper_->move(0.08, cmd.speed);
                        }
                        // Read back actual width and store atomically for broadcaster
                        franka::GripperState gs = gripper_->readOnce();
                        gripper_width_m_.store(gs.width);
                    } catch (const franka::Exception& e) {
                        std::cerr << "Gripper worker franka exception: " << e.what() << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Gripper worker std::exception: " << e.what() << std::endl;
                    }
                }
            });

            // --- Move to starting joint configuration ---
            std::array<double, 7> q_goal;
            if (bidexhand_) {
                q_goal = {{0.0, -0.812, -0.123, -2.0, 0.0, 2.8, 0.9}};  // BiDexHand pose
            } else {
                q_goal = {{0.0, -0.48,  0.0,   -2.0, 0.0, 1.57, -0.85}}; // Full range pose
            }
            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This example will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // --- Collision behavior ---
            robot.setCollisionBehavior(
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}});

            // --- Joint impedance ---
            robot.setJointImpedance({{3000,3000,3000,2500,2500,2000,2000}});

            // --- Read initial robot state ---
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));

            for (int i = 0; i < 7; i++) {
                current_joint_angles_[i] = state.q[i];
                neutral_joint_pose_[i]   = q_goal[i];
            }

            // --- IK solver ---
            std::array<double, 7> base_joint_weights = {{
                3.0,  // Joint 0 – base rotation:  high penalty for stability
                6.0,  // Joint 1 – base shoulder:  high penalty for stability
                1.5,  // Joint 2 – elbow
                1.5,  // Joint 3 – forearm
                1.0,  // Joint 4 – wrist
                1.0,  // Joint 5 – wrist
                1.0   // Joint 6 – hand
            }};
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_,
                1.0,   // manipulability weight
                2.0,   // neutral distance weight
                2.0,   // current distance weight
                base_joint_weights,
                false  // verbose = false for production use
            );

            // --- Ruckig ---
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i]     = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i]         = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i]     = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            std::cout << "Ruckig trajectory generator configured with 7 DOFs" << std::endl;

            // --- Initial VR targets = robot's starting pose ---
            vr_target_position_    = initial_robot_pose_.translation();
            vr_target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            // --- Start network thread and wait for VR data ---
            // While waiting, broadcast robot state so the Python data-collection
            // script can receive valid state immediately, independent of whether
            // the VR headset has sent its first packet yet.
            std::thread network_thread(&VRController::networkThread, this);
            broadcast_thread_ = std::thread(&VRController::broadcastThreadFunc, this);
            ik_thread_running_ = true;
            ik_thread_ = std::thread(&VRController::ikThreadFunc, this);

            std::cout << "Waiting for VR data..." << std::endl;
            while (!vr_initialized_ && running_) {
                // Read current robot state and broadcast it at ~10 Hz
                try {
                    franka::RobotState pre_state = robot.readOnce();
                    broadcastRobotState(makeSnapshot(pre_state));
                } catch (const franka::Exception& e) {
                    std::cerr << "Warning: readOnce failed during VR wait: " << e.what() << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (vr_initialized_) {
                std::cout << "VR initialized! Starting real-time control." << std::endl;
                this->runVRControl(robot);
            }

            running_ = false;
            ik_thread_running_ = false;
            if (network_thread.joinable())
                network_thread.join();
            if (broadcast_thread_.joinable())
                broadcast_thread_.join();
            if (ik_thread_.joinable())
                ik_thread_.join();

            gripper_thread_running_ = false;
            gripper_cv_.notify_one();
            if (gripper_thread_.joinable())
                gripper_thread_.join();
        }
        catch (const franka::Exception& e)
        {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    // -----------------------------------------------------------------------
    // runVRControl – real-time joint velocity control loop (1 kHz)
    // -----------------------------------------------------------------------
    void runVRControl(franka::Robot& robot)
    {
        auto vr_control_callback = [this](
            const franka::RobotState& robot_state,
            franka::Duration           period) -> franka::JointVelocities
        {
            // --- Update VR targets from latest command ---
            // try_lock: if the network thread is currently writing, keep the
            // previous command rather than blocking the RT callback.
            VRCommand cmd;
            if (command_mutex_.try_lock()) {
                cmd = current_vr_command_;
                command_mutex_.unlock();
            }
            // If try_lock fails, cmd.has_valid_data == false → updateVRTargets
            // returns early and Ruckig holds its previous targets for this cycle.
            updateVRTargets(cmd);

            // --- Gripper toggle on button press edge ---
            bool button_pressed = cmd.button_pressed > 0.5;
            if (button_pressed && !prev_button_pressed_) {
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close         = gripper_is_open_;
                    pending_gripper_cmd_.speed         = cmd.gripper_speed;
                    pending_gripper_cmd_.force         = cmd.gripper_force;
                    pending_gripper_cmd_.epsilon_inner = cmd.epsilon_inner;
                    pending_gripper_cmd_.epsilon_outer = cmd.epsilon_outer;
                    gripper_requested_ = true;
                }
                gripper_cv_.notify_one();
                gripper_is_open_ = !gripper_is_open_;
            }
            prev_button_pressed_ = button_pressed;

            // --- Ruckig initialization (first call only) ---
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i]            = robot_state.q[i];
                    ruckig_input_.current_position[i]    = robot_state.q[i];
                    ruckig_input_.current_velocity[i]    = 0.0;
                    ruckig_input_.current_acceleration[i]= 0.0;
                    ruckig_input_.target_position[i]     = robot_state.q[i];
                    ruckig_input_.target_velocity[i]     = 0.0;
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
                std::cout << "Ruckig initialized – starting with zero velocity commands." << std::endl;
            } else {
                // Use Ruckig's own previous output for continuity (avoid encoder noise)
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i]             = robot_state.q[i];
                    ruckig_input_.current_position[i]    = robot_state.q[i];
                    ruckig_input_.current_velocity[i]    = ruckig_output_.new_velocity[i];
                    ruckig_input_.current_acceleration[i]= ruckig_output_.new_acceleration[i];
                }
            }

            // --- Gradual activation factor ---
            auto   current_time   = std::chrono::steady_clock::now();
            double elapsed_sec    = std::chrono::duration<double>(current_time - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);

            // --- Post IK request to off-RT thread (~200 Hz) ---
            static int debug_counter = 0;
            debug_counter++;

            std::array<double, 3> target_pos = eigenToArray3(vr_target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(vr_target_orientation_);
            double current_q7 = current_joint_angles_[6];
            double q7_start   = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end     = std::min( 2.89, current_q7 + Q7_SEARCH_RANGE);

            if (debug_counter % 5 == 0) {
                if (ik_input_mutex_.try_lock()) {
                    ik_input_.target_pos     = target_pos;
                    ik_input_.target_rot     = target_rot;
                    ik_input_.current_joints = current_joint_angles_;
                    ik_input_.q7_start       = q7_start;
                    ik_input_.q7_end         = q7_end;
                    ik_input_ready_.store(true, std::memory_order_release);
                    ik_input_mutex_.unlock();
                }
            }

            // --- Read latest IK result (non-blocking; fall back to cache) ---
            if (ik_output_mutex_.try_lock()) {
                if (ik_output_valid_) {
                    cached_ik_result_ = ik_output_;
                    cached_ik_valid_  = true;
                }
                ik_output_mutex_.unlock();
            }
            WeightedIKResult ik_result;
            ik_result.success = cached_ik_valid_;
            if (cached_ik_valid_) {
                ik_result = cached_ik_result_;
            }

            // --- Debug printout (every 1 s) — minimized to reduce syscall frequency ---
            if (debug_counter % 1000 == 0) {
                std::cout << "IK: "
                          << (cached_ik_valid_ ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m")
                          << " | Joints: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(2) << current_joint_angles_[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << '\n';  // avoid endl flush inside RT callback
            }

            // --- Update Ruckig targets ---
            if (ik_result.success) {
                for (int i = 0; i < 7; i++) {
                    double current_pos    = current_joint_angles_[i];
                    double ik_target_pos  = ik_result.joint_angles[i];
                    ruckig_input_.target_position[i] =
                        current_pos + activation_factor * (ik_target_pos - current_pos);
                    ruckig_input_.target_velocity[i] = 0.0;
                }
                // Enforce BiDexHand / full-range q7 limits
                ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
            }
            // On IK failure: keep previous targets (Ruckig will hold/decelerate)

            // --- Ruckig update ---
            ruckig::Result ruckig_result =
                trajectory_generator_->update(ruckig_input_, ruckig_output_);

            std::array<double, 7> target_joint_velocities;
            if (ruckig_result == ruckig::Result::Working ||
                ruckig_result == ruckig::Result::Finished) {
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency stop
                target_joint_velocities.fill(0.0);
                if (debug_counter % 1000 == 0) {
                    std::cout << "Ruckig error – zero velocity for safety.\n";
                }
            }

            // --- Capture state snapshot for off-RT broadcast thread (~100 Hz) ---
            // try_lock: if the broadcast thread holds the mutex, skip this cycle
            // rather than blocking the RT callback.
            if (debug_counter % 10 == 0) {
                if (broadcast_mutex_.try_lock()) {
                    broadcast_snapshot_ = makeSnapshot(robot_state);
                    broadcast_snapshot_ready_.store(true, std::memory_order_release);
                    broadcast_mutex_.unlock();
                }
            }

            if (!running_) {
                return franka::MotionFinished(
                    franka::JointVelocities({0.0,0.0,0.0,0.0,0.0,0.0,0.0}));
            }
            return franka::JointVelocities(target_joint_velocities);
        };

        try {
            robot.control(vr_control_callback);
        } catch (const franka::ControlException& e) {
            std::cerr << "VR control exception: " << e.what() << std::endl;
        }
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname> [bidexhand]" << std::endl;
        std::cerr << "  bidexhand: true (default) for BiDexHand limits, false for full range"
                  << std::endl;
        return -1;
    }

    bool bidexhand = false;
    if (argc == 3) {
        std::string arg = argv[2];
        bidexhand = (arg == "true" || arg == "1");
    }

    try {
        VRController controller(bidexhand);
        controller.run(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}