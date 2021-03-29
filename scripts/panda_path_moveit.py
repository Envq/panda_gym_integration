#!/usr/bin/env python3

# ROS and Moveit
import rospy
import moveit_commander
from rospy.rostime import Duration

# Other
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../panda_controller/scripts/src")))

# panda_controller
from panda_interface_moveit import PandaInterfaceMoveit



def parseLine(line):
    target = list()
    for e in line.split():
        target.append(float(e))
    return (target[:7], target[7])


def main():
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_path_node', anonymous=True)

        # Get roslaunch parameters
        mode = rospy.get_param('~mode', 1)
        mode2_delay = rospy.get_param('~mode2_delay', 1.0)
        mode2_wait = rospy.get_param('~mode2_wait', False)
        file_name = rospy.get_param('~file_name', "path_test")
        real_robot = rospy.get_param('~real_robot', False)
        arm_speed = rospy.get_param('~arm_speed', 0.1)

        gripper_speed = rospy.get_param('~gripper_speed', 0.1)
        grasp_option = dict()
        grasp_option['speed'] = rospy.get_param('~grasp_speed', 0.1)
        grasp_option['force'] = rospy.get_param('~grasp_force', 10.0)
        grasp_option['epsilon_inner'] = rospy.get_param('~grasp_epsilon_inner', 0.02)
        grasp_option['epsilon_outer'] = rospy.get_param('~grasp_epsilon_outer', 0.02)

        eef_step = rospy.get_param('~eef_step', 0.01)
        jump_threashould = rospy.get_param('~jump_threashould', 0.00)


        # Create panda moveit interface
        print(real_robot)
        panda = PandaInterfaceMoveit(\
                            delay=1,\
                            arm_velocity_factor=arm_speed,\
                            startup_homing=False,\
                            real_robot=real_robot)
        
        # Reading file
        FILE_PATH_NAME = os.path.join(os.path.dirname(__file__), "../data/paths/" + file_name + ".txt")
        file_reader = open(FILE_PATH_NAME, 'r')
        start = True
        last_gripper = 0.08
        waypoints = list()


        step = 1
        for line in file_reader:
            (pose, gripper) = parseLine(line)
            grasp = 0 if gripper == 0.08 else 1
            print("[step {}]: {} and {}".format(step, pose, gripper))
            step += 1

            if start:
                print("Go to Start Pose...")
                print("Success: ", panda.movePose(pose + [gripper, 0], wait_execution=True))
                start = False
            
            else:
                if mode == 1:
                    waypoints.append(pose)
                elif mode == 2:
                    print("Success: ", panda.moveArmPoseTCP(pose, wait_execution=mode2_wait))
                    rospy.sleep(Duration(mode2_delay))

                if gripper != last_gripper:
                    if mode == 1:
                        panda.execute_tcp_cartesian_path(waypoints, eef_step, jump_threashould)
                        waypoints = list()  # clear waypoints list
                    if grasp:
                        panda.graspGripper(gripper, **grasp_option)
                    else:
                        panda.moveGripper(gripper, speed=gripper_speed)
                last_gripper = gripper

        # Always open gripper to the end
        panda.moveGripper(0.08, speed=gripper_speed)

    except rospy.ROSInterruptException:
        print("ROS interrupted")
    
    # Close file
    file_reader.close()



if __name__ == "__main__":
    main()