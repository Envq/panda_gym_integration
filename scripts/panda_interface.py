#!/usr/bin/env python2

from src.panda_client import panda_interface
import sys, rospy, moveit_commander


try:
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_interface', anonymous=True)

    # Launch panda interface
    interface = panda_interface(HOST, PORT)

    while True:
        interface.sendCurrentJoints()
        response = interface.getResponse()
        if response == 'close':
            break
        print("Success? ", response)

except rospy.ROSInterruptException:
    print("ROS interrupted")
except KeyboardInterrupt:
    print("Keboard quit")
