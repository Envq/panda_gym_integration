#!/usr/bin/env python2

from src.panda_moveit import PandaMoveitInterface
import sys, rospy, moveit_commander
from geometry_msgs.msg import Pose
import copy
import os



def getWaypoint(panda, pose):
    wrist_to_target = panda.getWristFromTCP(pose)
    waypoint = Pose()
    waypoint.position.x = wrist_to_target[0]
    waypoint.position.y = wrist_to_target[1]
    waypoint.position.z = wrist_to_target[2]
    waypoint.orientation.x = wrist_to_target[3]
    waypoint.orientation.y = wrist_to_target[4]
    waypoint.orientation.z = wrist_to_target[5]
    waypoint.orientation.w = wrist_to_target[6]
    return waypoint


def parseLine(line):
    # Get target_pose and transform tcp->gym
    target = list()
    for e in line.split():
        target.append(float(e))
    return (target[:7], target[7])


def main(FILE_PATH):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_trajectory', anonymous=True)

        # Get roslaunch parameters
        use_real_robot = rospy.get_param('~use_real_robot', False)

        grasp_option = dict()
        gripper_speed = grasp_option['speed'] = rospy.get_param('~gripper_speed', 0.1)
        grasp_option['force'] = rospy.get_param('~grasp_force', 10.0)
        grasp_option['epsilon_inner'] = rospy.get_param('~grasp_epsilon_inner', 0.02)
        grasp_option['epsilon_outer'] = rospy.get_param('~grasp_epsilon_outer', 0.02)

        eef_step = rospy.get_param('~eef_step', 0.01)
        jump_threashould = rospy.get_param('~jump_threashould', 0.00)

        # Create panda moveit interface
        panda = PandaMoveitInterface(delay=1, real_robot=use_real_robot)
        
        # Reading file
        file_reader = open(FILE_PATH, 'r')
        start = True
        last_gripper = 0.08
        waypoints = list()

        for line in file_reader:
            (pose, gripper) = parseLine(line)
            grasp = 0 if gripper == 0.08 else 1

            if start:
                print("Go to Start Pose...")
                panda.movePose(pose + [gripper, 0])
                panda.moveArmPoseTCP(pose)
                panda.moveGripper(gripper, speed=gripper_speed)
                start = False
            
            else:
                waypoints.append(getWaypoint(panda, pose))
                if gripper != last_gripper:
                    panda.execute_cartesian_path(waypoints, eef_step, jump_threashould)
                    # clear waypoints list
                    waypoints = list()
                    if grasp:
                        panda.graspGripper(gripper, **grasp_option)
                    else:
                        panda.moveGripper(gripper, speed=gripper_speed)
                last_gripper = gripper


    except rospy.ROSInterruptException:
        print("ROS interrupted")
    
    # Close file
    file_reader.close()



if __name__ == "__main__":
    FILE_NAME = "path_test"


    file_path = os.path.join(os.path.dirname(__file__), "../data/paths/" + FILE_NAME + ".txt")
    main(file_path)