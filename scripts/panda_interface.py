#!/usr/bin/env python2

from src.panda_client import GymInterface
from src.panda_moveit import PandaMoveitInterface
import sys, rospy, moveit_commander



def main(HOST, PORT):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_interface', anonymous=True)

        # Create gym interface
        gym = GymInterface(HOST, PORT)

        # Create panda moveit interface
        panda = PandaMoveitInterface(delay=1)

        while True:
            # Get current pose
            current_pose = panda.getPose()

            # Send current pose
            gym.sendCurrentState(current_pose)

            # Get goal pose
            goal_pose = gym.getGoalState()
            
            # Check close
            if goal_pose == 'close':
                break

            # Run goal joints
            success = panda.moveToPose(goal_pose)
            print("Success? ", success)
            if not success:
                print("Error! abort")
                gym.sendError()
                break

    except rospy.ROSInterruptException:
        print("ROS interrupted")



if __name__ == "__main__":
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    main(HOST, PORT)