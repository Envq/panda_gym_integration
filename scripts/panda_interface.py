#!/usr/bin/env python2

from src.panda_client import GymInterface
import sys, rospy, moveit_commander


try:
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_interface', anonymous=True)

    # Launch panda interface
    interface = GymInterface(HOST, PORT)

    while True:
        # Send current joints
        interface.sendCurrentJoints()

        # Get goal joints
        goal_joints = interface.getGoalJoints()
        
        # Check close
        if goal_joints == 'close':
            break

        # Run goal joints
        success = interface.movePandaTo(goal_joints)
        print("Success? ", success)
        if not success:
            print("Error! abort")
            interface.sendError()
            break

except rospy.ROSInterruptException:
    print("ROS interrupted")
except KeyboardInterrupt:
    print("Keboard quit")
