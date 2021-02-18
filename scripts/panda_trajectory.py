#!/usr/bin/env python2

from src.panda_moveit import PandaMoveitInterface
import sys, rospy, moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import copy
import os


def main(FILE_PATH):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_trajectory', anonymous=True)

        # Create panda moveit interface
        panda = PandaMoveitInterface(delay=1)

        # Create waypoints
        waypoints = list()
        with open(FILE_PATH, 'r') as file_reader:
            # Get waipoints
            for line in file_reader:
                # Get target_pose and transform tcp->gym
                target_pose = list()
                for e in line.split():
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
                waypoints.append(copy.deepcopy(waypoint))

        # Execute cartesian path from this waypoints
        panda.execute_cartesian_path(waypoints=waypoints)

    except rospy.ROSInterruptException:
        print("ROS interrupted")



if __name__ == "__main__":
    FILE_NAME = "trajectory"


    file_path = os.path.join(os.path.dirname(__file__), "../data/trajectories/" + FILE_NAME + ".txt")
    main(file_path)