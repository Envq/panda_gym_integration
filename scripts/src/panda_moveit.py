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
    self.arm = moveit_commander.MoveGroupCommander("panda_arm")
    self.hand = moveit_commander.MoveGroupCommander("hand")
  

  def getArmJoints(self):
    """[j0, j1, j2, j3, j4, j5, j6, j7]"""
    return self.arm.get_current_joint_values()


  def getHandJoints(self):
    """[j8, j9]"""
    return self.hand.get_current_joint_values()


  def getArmPose(self):
    """[px, py, pz, ox, oy, oz, ow]"""
    pose = self.arm.get_current_pose().pose
    l = list()
    l.append(pose.position.x)
    l.append(pose.position.y)
    l.append(pose.position.z)
    l.append(pose.orientation.x)
    l.append(pose.orientation.y)
    l.append(pose.orientation.z)
    l.append(pose.orientation.w)
    return l


  def getJoints(self):
    """[j0, j1, j2, j3, j4, j5, j6, j7, j8, j9]"""
    return self.getArmJoints() + self.getHandJoints()
  

  def getPose(self):
    """[px, py, pz, ox, oy, oz, ow, gripper_size]"""
    return self.getArmPose() + [self.getHandJoints()[0]*2]
  

  def moveToArmJoints(self, arm_goal_joints):
    if len(arm_goal_joints) != 7:
      return False
    try:
      self.arm.plan(arm_goal_joints)
      self.arm.go(wait=True)
    except MoveItCommanderException:
      return False
    return True
  

  def moveToHandJoints(self, hand_goal_joints):
    if len(hand_goal_joints) != 2:
      return False
    try:
      self.hand.plan(hand_goal_joints)
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  def moveToJoints(self, goal_joints):
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
    return self.moveToArmJoints((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


  def moveToArmPose(self, goal_pose):
    if len(goal_pose) != 7:
      return False
    target = geometry_msgs.msg.Pose()
    target.position.x = goal_pose[0]
    target.position.y = goal_pose[1]
    target.position.z = goal_pose[2]
    target.orientation.x = goal_pose[3]
    target.orientation.y = goal_pose[4]
    target.orientation.z = goal_pose[5]
    target.orientation.w = goal_pose[6]
    try:
      self.arm.plan(target)
      self.arm.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  def moveToPose(self, goal_pose):
    if len(goal_pose) != 8:
      return False
    target = geometry_msgs.msg.Pose()
    target.position.x = goal_pose[0]
    target.position.y = goal_pose[1]
    target.position.z = goal_pose[2]
    target.orientation.x = goal_pose[3]
    target.orientation.y = goal_pose[4]
    target.orientation.z = goal_pose[5]
    target.orientation.w = goal_pose[6]
    gripper = [goal_pose[7]/2.0, goal_pose[7]/2.0]
    try:
      self.arm.plan(target)
      self.hand.plan(gripper)
      self.arm.go(wait=True)
      self.hand.go(wait=True)
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
    print("--------------------")
    print(panda.getHandJoints())
    print("--------------------")
    print(panda.getJoints())
    print("--------------------")
    print(panda.getArmPose())
    print("--------------------")
    print(panda.getPose())
    print("--------------------")

    # Perform custom commands
    while True:
      code = input("Insert code: ")
      if (code == '0'):
        sys.exit()
      elif (code == '1'):
        print("Success? ", panda.moveToReadyPose())
      elif (code == '2'):
        print("Success? ", panda.moveToArmJoints((0, -pi/4, 0, -pi/2, 0, pi/3, 0)))
      elif (code == '3'):
        # Fail test
        print("Success? ", panda.moveToArmJoints((0, 0, 0, 0, 0, 0, 0))) 
      elif (code == '4'):
        print("Success? ", panda.moveToHandJoints((0.0, 0.0)))
      elif (code == '5'):
        print("Success? ", panda.moveToHandJoints((0.01, 0.01)))
      elif (code == '6'):
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.0, 0.0)))
      elif (code == '7'):
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.01, 0.01)))
      elif (code == '8'):
        # Fail test
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.1, 0.1))) 
      elif (code == '9'):
        print("Success? ", panda.moveToArmPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0)))
      elif (code == '10'):
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0, 0.06)))
      elif (code == '11'):
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0, 0.01)))
      elif (code == '12'):
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0, 0.00)))
      elif (code == '13'):
        # Fail test
        print("Success? ", panda.moveToPose((0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0, 0.1)))
      else:
        print("Code not valid")

  except rospy.ROSInterruptException:
    print("ROS interrupted")
    sys.exit()
  except KeyboardInterrupt:
    print("Keboard quit")
    sys.exit()
    