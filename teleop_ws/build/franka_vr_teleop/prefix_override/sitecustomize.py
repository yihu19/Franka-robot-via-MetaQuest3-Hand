import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hca_research/franka_teleoperation_collection/meta_quest/franka_research/teleop_ws/install/franka_vr_teleop'
