#!/usr/bin/env python3
"""
Receiver script to parse pose data sent from the ROS pose_socket_sender node.

This script listens on a UDP socket and decodes the binary pose data.
"""

import socket
import struct
import sys


def main():
    # Configuration
    listen_ip = '192.168.0.214'  # Listen on all interfaces
    listen_port = 9000      # Default port (change as needed)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        listen_port = int(sys.argv[1])
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_ip, listen_port))
    
    print(f"Pose Receiver listening on {listen_ip}:{listen_port}")
    print("Waiting for pose data...")
    print("=" * 80)
    
    try:
        while True:
            # Receive data
            data, addr = sock.recvfrom(1024)
            
            # Check if we received the expected amount of data (7 doubles = 56 bytes)
            if len(data) == 64:
                # Unpack 7 doubles in little-endian format
                pose_data = struct.unpack('<8d', data)
                
                # Extract position and quaternion
                pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, press_index = pose_data
                
                # Display the received data
                print(f"\nReceived from {addr[0]}:{addr[1]}")
                print(f"Position:")
                print(f"  x: {pos_x:10.6f}")
                print(f"  y: {pos_y:10.6f}")
                print(f"  z: {pos_z:10.6f}")
                print(f"Quaternion:")
                print(f"  x: {quat_x:10.6f}")
                print(f"  y: {quat_y:10.6f}")
                print(f"  z: {quat_z:10.6f}")
                print(f"  w: {quat_w:10.6f}")
                print(f"Press Index: {press_index:10.6f}")
                print("-" * 80)
            else:
                print(f"Warning: Received {len(data)} bytes, expected 64 bytes")
    
    except KeyboardInterrupt:
        print("\n\nShutting down receiver...")
        sock.close()
        sys.exit(0)


if __name__ == '__main__':
    main()
