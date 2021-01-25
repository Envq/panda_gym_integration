#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

from moveit_commander.exception import MoveItCommanderException
import moveit_commander
import geometry_msgs.msg
from math import pi



class MoveGroupInterface(object):
  def __init__(self):
    super(MoveGroupInterface, self).__init__()
    self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
  

  def getJoints(self):
    return self.move_group.get_current_joint_values()
  

  def getPose(self):
    return self.move_group.get_current_pose()


  def moveToJoints(self, goal_joints):
    if len(goal_joints) != 7:
      return False
    try:
      self.move_group.go(goal_joints, wait=True)
    except MoveItCommanderException:
      return False
    return True
  
  
  def moveToReadyPose(self):
    return self.moveToJoints((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


  def moveToPose(self, goal_pose):
    if len(goal_pose) != 7:
      return False
    target = geometry_msgs.msg.Pose()
    target.position.x = goal_pose[0]
    target.position.y = goal_pose[1]
    target.position.z = goal_pose[2]
    target.orientation.w = goal_pose[3]
    target.orientation.x = goal_pose[4]
    target.orientation.y = goal_pose[5]
    target.orientation.z = goal_pose[6]
    try:
      self.move_group.set_pose_target(target)
      plan = self.move_group.go(wait=True)
      self.move_group.clear_pose_targets()
    except MoveItCommanderException:
      return False
    return True



if __name__ == '__main__':
  """TEST"""
  import sys, rospy

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
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0)))
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
    