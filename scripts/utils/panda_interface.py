#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import moveit_commander
from moveit_commander.exception import MoveItCommanderException
import geometry_msgs.msg
from math import pi


class MoveGroupInterface(object):
  """MoveGroupPythonInteface"""
  def __init__(self):
    super(MoveGroupInterface, self).__init__()

    ## This interface can be used to plan and execute motions:
    self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
  

  def getJoints(self):
    return self.move_group.get_current_joint_values()
  

  def getPose(self):
    return self.move_group.get_current_pose()


  def moveToJoints(self, joint_goal):
    if len(joint_goal) != 7:
      return False
    try:
      self.move_group.go(joint_goal, wait=True)
    except MoveItCommanderException:
      return False
    return True
  
  
  def moveToReadyPose(self):
    return self.moveToJoints((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


  def moveToPose(self, pose_goal):
    if len(pose_goal) != 7:
      return False
    target = geometry_msgs.msg.Pose()
    target.position.x = pose_goal[0]
    target.position.y = pose_goal[1]
    target.position.z = pose_goal[2]
    target.orientation.w = pose_goal[3]
    target.orientation.x = pose_goal[4]
    target.orientation.y = pose_goal[5]
    target.orientation.z = pose_goal[6]
    try:
      self.move_group.set_pose_target(target)
      plan = self.move_group.go(wait=True)
      self.move_group.clear_pose_targets()
    except MoveItCommanderException:
      return False
    return True

