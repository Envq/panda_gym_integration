#!/usr/bin/env python2

from utils.panda_interface import *
import sys, rospy


if __name__ == '__main__':
  try:
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

    # Inizialize movegroupinterface
    panda = MoveGroupInterface()

    while True:
      code = input("Insert code: ")
      if (code == '0'):
        sys.exit()
      elif (code == '1'):
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0, 0, 0)))
      elif (code == '2'):
        print("Success? ", panda.moveToReadyPose())
      elif (code == '3'):
        print("Success? ", panda.moveToJoints((0, -pi/4, 0, -pi/2, 0, pi/3, 0)))
      else:
        print("Code not valid")

  except rospy.ROSInterruptException:
    print("ROS interrupted")
    sys.exit()
  except KeyboardInterrupt:
    print("Keboard quit")
    sys.exit()
    