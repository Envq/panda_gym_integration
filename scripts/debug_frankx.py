from frankx import *
from math import pi
import sys

if __name__ == '__main__':
    robot = Robot("192.168.1.2")
    gripper = robot.get_gripper()
    robot.set_default_behavior()
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.set_dynamic_rel(0.05)

    cmd = sys.argv[1]

    if cmd == 'read':
        current = robot.current_pose()
        print('Pose: ', current.vector().tolist())

    elif cmd == 'ready':
        # Homing
        robot.move(JointMotion([0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi]))

    elif cmd == 'relative':
        # Define and move forwards
        way = Affine(0.00, 0.00, 0.05)
        motion_forward = LinearRelativeMotion(way)
        robot.move(motion_forward)
    
    elif cmd == 'absolute':
        pose1 = Affine(0.618024652107368, -0.018727033651026792, 0.125)
        pose2 = Affine(0.4898394521073681, -0.07501663365102673, 0.20)
        motion_pose = LinearMotion(pose2)
        robot.move(motion_pose)
        
    elif cmd == 'g_close':
        gripper.move(0.0)
    
    elif cmd == 'g_open':
        gripper.move(0.08)
    
    elif cmd == 'g_val':
        gripper.move(0.04)
    
    elif cmd == 'g_clamp':
        gripper.clamp()
    
    elif cmd == 'gripper':
        print("width: ", gripper.width())
        print("is_grasping", gripper.is_grasping())
        print("max: ", gripper.max_width)


    else:
        print("command not valid")


