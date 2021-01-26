#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

from moveit_commander.exception import MoveItCommanderException
import moveit_commander
import geometry_msgs.msg
from moveit_msgs.msg import RobotTrajectory
from math import pi



class MoveGroupInterface(object):
  def __init__(self):
    super(MoveGroupInterface, self).__init__()
    self.arm = moveit_commander.MoveGroupCommander("panda_arm")
    self.hand = moveit_commander.MoveGroupCommander("hand")
  

  def getArmJoints(self):
    return self.arm.get_current_joint_values()


  def getHandJoints(self):
    return self.hand.get_current_joint_values()


  def getJoints(self):
    return self.getArmJoints() + self.getHandJoints()
  

  def getPose(self):
    return self.arm.get_current_pose()
  

  def moveArmJointsTo(self, arm_goal_joints):
    if len(arm_goal_joints) != 7:
      return False
    try:
      self.arm.plan(arm_goal_joints)
      self.arm.go(wait=True)
    except MoveItCommanderException:
      return False
    return True
  

  def moveHandJointsTo(self, hand_goal_joints):
    if len(hand_goal_joints) != 2:
      return False
    try:
      self.hand.plan(hand_goal_joints)
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  def moveJointsTo(self, goal_joints):
    if len(goal_joints) != 9:
      return False
    try:
      self.arm.plan(goal_joints[:7])
      self.hand.plan(goal_joints[7:9])
      self.arm.go(wait=True)
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True
  
  
  def moveToReadyPose(self):
    return self.moveArmJointsTo((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


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
      self.arm.set_pose_target(target)
      plan = self.arm.go(wait=True)
      self.arm.clear_pose_targets()
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

    # Print Status
    print(panda.getArmJoints())
    print(panda.getHandJoints())
    print(panda.getJoints())

    # Perform custom commands
    while True:
      code = input("Insert code: ")
      if (code == '0'):
        sys.exit()
      elif (code == '1'):
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0)))
      elif (code == '2'):
        print("Success? ", panda.moveToReadyPose())
      elif (code == '3'):
        print("Success? ", panda.moveArmJointsTo((0, -pi/4, 0, -pi/2, 0, pi/3, 0)))
      elif (code == '4'):
        # Fail test
        print("Success? ", panda.moveArmJointsTo((0, 0, 0, 0, 0, 0, 0))) 
      elif (code == '5'):
        print("Success? ", panda.moveHandJointsTo((0.0, 0.0)))
      elif (code == '6'):
        print("Success? ", panda.moveHandJointsTo((0.01, 0.01)))
      elif (code == '7'):
        print("Success? ", panda.moveJointsTo((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.0, 0.0)))
      elif (code == '8'):
        print("Success? ", panda.moveJointsTo((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.01, 0.01)))
      elif (code == '9'):
        # Fail test
        print("Success? ", panda.moveJointsTo((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.1, 0.1))) 
      else:
        print("Code not valid")

  except rospy.ROSInterruptException:
    print("ROS interrupted")
    sys.exit()
  except KeyboardInterrupt:
    print("Keboard quit")
    sys.exit()
    