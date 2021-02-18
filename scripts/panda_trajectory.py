#!/usr/bin/env python2

from src.panda_moveit import PandaMoveitInterface
import sys, rospy, moveit_commander
from geometry_msgs.msg import Pose
import copy
import os


def generateWaypointFromLine(line, panda):
    # Get target_pose and transform tcp->gym
    target_pose = list()
    for e in line:
        target_pose.append(float(e))
    target_wrist = panda.getWristFromTCP(target_pose[:7])

    # Generate waypoint and add it
    waypoint = Pose()
    waypoint.position.x = target_wrist[0]
    waypoint.position.y = target_wrist[1]
    waypoint.position.z = target_wrist[2]
    waypoint.orientation.x = target_wrist[3]
    waypoint.orientation.y = target_wrist[4]
    waypoint.orientation.z = target_wrist[5]
    waypoint.orientation.w = target_wrist[6]
    return waypoint


def main(FILE_PATH):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_trajectory', anonymous=True)

        # Create panda moveit interface
        panda = PandaMoveitInterface(delay=1)

        # Read trajectory file
        with open(FILE_PATH, 'r') as file_reader:
            # GRIPPER OPEN
            line = file_reader.readline()
            gripper = float(line)
            panda.moveToHandPose(gripper)


            # START
            line = file_reader.readline()
            tcp_target_pose = list()
            for e in line.split():
                tcp_target_pose.append(float(e))
            panda.moveToArmPoseTCP(tcp_target_pose)


            # PRE-GRASP and GRASP
            waypoints = list()
            while True:
                line = file_reader.readline()
                components = line.split()
                
                if len(components) == 1:
                    break

                waypoint = generateWaypointFromLine(components, panda)
                waypoints.append(copy.deepcopy(waypoint))
            panda.execute_cartesian_path(waypoints=waypoints)


            # CLOSE GRIPPER
            gripper = float(line)
            panda.moveToHandPose(gripper)


            # PLACE
            waypoints = list()
            while True:
                line = file_reader.readline()
                components = line.split()
                
                if len(components) == 1:
                    break

                waypoint = generateWaypointFromLine(components, panda)
                waypoints.append(copy.deepcopy(waypoint))
            panda.execute_cartesian_path(waypoints=waypoints)


            # OPEN GRIPPER
            gripper = float(line)
            panda.moveToHandPose(gripper)


    except rospy.ROSInterruptException:
        print("ROS interrupted")



if __name__ == "__main__":
    FILE_NAME = "trajectory"


    file_path = os.path.join(os.path.dirname(__file__), "../data/trajectories/" + FILE_NAME + ".txt")
    main(file_path)