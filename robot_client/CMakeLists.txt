cmake_minimum_required(VERSION 3.10)
project(franka_vr_control_client)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization level to Release
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set CMAKE_PREFIX_PATH to find your libfranka installation
# You can override this by passing -DFRANKA_INSTALL_PATH=/path/to/franka/install
set(FRANKA_INSTALL_PATH "$ENV{HOME}/ws/franka/install" CACHE PATH "Path to Franka installation directory")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FRANKA_INSTALL_PATH}")

# Find required packages
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
# Prefer config package; fall back to system header/library lookup.
find_package(ruckig QUIET)
if(NOT ruckig_FOUND)
    find_path(RUCKIG_INCLUDE_DIR NAMES ruckig/ruckig.hpp
        PATHS
            /usr/include
            /usr/local/include
            $ENV{HOME}/ws/ruckig/install/include
            $ENV{HOME}/ws/ruckig/include
            $ENV{HOME}/franky/ruckig/include)
    if(NOT RUCKIG_INCLUDE_DIR AND EXISTS "/usr/include/ruckig/ruckig.hpp")
        set(RUCKIG_INCLUDE_DIR "/usr/include")
    endif()
    if(NOT RUCKIG_INCLUDE_DIR AND EXISTS "/usr/local/include/ruckig/ruckig.hpp")
        set(RUCKIG_INCLUDE_DIR "/usr/local/include")
    endif()
    if(NOT RUCKIG_INCLUDE_DIR AND EXISTS "$ENV{HOME}/ws/ruckig/install/include/ruckig/ruckig.hpp")
        set(RUCKIG_INCLUDE_DIR "$ENV{HOME}/ws/ruckig/install/include")
    endif()
    if(NOT RUCKIG_INCLUDE_DIR AND EXISTS "$ENV{HOME}/ws/ruckig/include/ruckig/ruckig.hpp")
        set(RUCKIG_INCLUDE_DIR "$ENV{HOME}/ws/ruckig/include")
    endif()
    if(NOT RUCKIG_INCLUDE_DIR AND EXISTS "$ENV{HOME}/franky/ruckig/include/ruckig/ruckig.hpp")
        set(RUCKIG_INCLUDE_DIR "$ENV{HOME}/franky/ruckig/include")
    endif()
    find_library(RUCKIG_LIBRARY NAMES ruckig
        PATHS
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
            $ENV{HOME}/ws/ruckig/install/lib
            $ENV{HOME}/ws/ruckig/build)
    if(RUCKIG_INCLUDE_DIR)
        if(RUCKIG_LIBRARY)
            add_library(ruckig::ruckig UNKNOWN IMPORTED)
            set_target_properties(ruckig::ruckig PROPERTIES
                IMPORTED_LOCATION "${RUCKIG_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${RUCKIG_INCLUDE_DIR}"
            )
            message(STATUS "Found ruckig via manual lookup: ${RUCKIG_LIBRARY}")
        else()
            # Community builds can be header-only.
            add_library(ruckig::ruckig INTERFACE IMPORTED)
            set_target_properties(ruckig::ruckig PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${RUCKIG_INCLUDE_DIR}"
            )
            message(STATUS "Found header-only ruckig at: ${RUCKIG_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR "Could not find ruckig package or system library/header")
    endif()
endif()

# Add libraries
add_library(examples_common src/examples_common.cpp)
target_link_libraries(examples_common Franka::Franka)
target_include_directories(examples_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})

# Add geofik library
add_library(geofik src/geofik.cpp)
target_include_directories(geofik PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})

# Add weighted_ik library
add_library(weighted_ik src/weighted_ik.cpp)
target_link_libraries(weighted_ik geofik)
target_include_directories(weighted_ik PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})

# Add VR Franka control executable
add_executable(franka_vr_control_client src/franka_vr_control_client.cpp)

# Link libraries
target_link_libraries(franka_vr_control_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

# Include directories
target_include_directories(franka_vr_control_client PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Optional: Set runtime path for shared libraries
set_target_properties(franka_vr_control_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add record client executable (records robot state)
add_executable(franka_vr_control_record_client src/franka_vr_control_record_client.cpp)

target_link_libraries(franka_vr_control_record_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_vr_control_record_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_vr_control_record_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
# Add data sender client executable (sends robot state via UDP)
add_executable(franka_vr_control_data_sender_client src/franka_vr_control_data_sender_client.cpp)

target_link_libraries(franka_vr_control_data_sender_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_vr_control_data_sender_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Add RT-optimized data sender client executable
add_executable(franka_vr_control_rt_client src/franka_vr_control_rt_client.cpp)

target_link_libraries(franka_vr_control_rt_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_vr_control_rt_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_vr_control_rt_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

set_target_properties(franka_vr_control_data_sender_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add client that uses current pose as initial pose (no move to start)
add_executable(franka_vr_control_current_pose_client src/franka_vr_control_current_pose_client.cpp)

target_link_libraries(franka_vr_control_current_pose_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_vr_control_current_pose_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_vr_control_current_pose_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add Recover client executable
add_executable(franka_vr_control_recover_client src/franka_vr_control_recover_client.cpp)

target_link_libraries(franka_vr_control_recover_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_vr_control_recover_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_vr_control_recover_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# FR3 Robot Control Server (command/state bridge for Python client)
add_executable(franka_r3_robot_server src/franka_r3_robot_server.cpp)

target_link_libraries(franka_r3_robot_server
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_r3_robot_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_r3_robot_server PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# FR3 pose command client (non-VR protocol)
add_executable(franka_pose_cmd_client src/franka_pose_cmd_client.cpp)

target_link_libraries(franka_pose_cmd_client
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

target_include_directories(franka_pose_cmd_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(franka_pose_cmd_client PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
