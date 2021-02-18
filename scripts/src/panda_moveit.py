#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

# Ros and Moveit
import rospy
from moveit_commander.exception import MoveItCommanderException
import moveit_commander

# TF2
import tf2_ros
from tf.transformations import quaternion_multiply
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

# Other
from math import pi
import time



class PandaMoveitInterface(object):
  def __init__(self, delay = 0):
    super(PandaMoveitInterface, self).__init__()
    self.arm = moveit_commander.MoveGroupCommander("panda_arm")
    self.hand = moveit_commander.MoveGroupCommander("hand")
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 

    # Wait for correct loading
    time.sleep(delay) 
    

  # JOINTS------------------------------------------------------------
  def getArmJoints(self):
    """[j0, j1, j2, j3, j4, j5, j6]"""
    return self.arm.get_current_joint_values()


  def getHandJoints(self):
    """[fj0 fj1]"""
    return self.hand.get_current_joint_values()


  def getJoints(self):
    """[j0, j1, j2, j3, j4, j5, j6, fj0 fj1]"""
    return self.getArmJoints() + self.getHandJoints()


  def moveToArmJoints(self, arm_goal_joints):
    """[j0, j1, j2, j3, j4, j5, j6]"""
    if len(arm_goal_joints) != 7:
      return False
    try:
      self.arm.plan(arm_goal_joints)
      self.arm.go(wait=True)
    except MoveItCommanderException:
      return False
    return True
  

  def moveToHandJoints(self, finger_goal_joint):
    """[fj] There is only one finger because they must be equal"""
    try:
      self.hand.plan([finger_goal_joint, finger_goal_joint])
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  def moveToJoints(self, goal_joints):
    """[j0, j1, j2, j3, j4, j5, j6, fj] There is only one finger because they must be equal"""
    if len(goal_joints) != 8:
      return False
    try:
      self.arm.plan(goal_joints[:7])
      self.hand.plan([goal_joints[7], goal_joints[7]])
      self.arm.go(wait=True)
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  def moveToReady(self):
    return self.moveToArmJoints((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


  # HAND POSE---------------------------------------------------------
  def getHandPose(self):
    """[fd] fd is the distance beetween fingers"""
    return [self.getHandJoints()[0]*2]


  def moveToHandPose(self, goal_pose):
    """[fd] fd is the distance beetween fingers"""
    return self.moveToHandJoints(goal_pose / 2.0)


  # WRIST POSE--------------------------------------------------------
  def getArmPoseWrist(self):
    """[px, py, pz, ox, oy, oz, ow]"""
    pose = self.arm.get_current_pose().pose
    wrist = list()
    wrist.append(pose.position.x)
    wrist.append(pose.position.y)
    wrist.append(pose.position.z)
    wrist.append(pose.orientation.x)
    wrist.append(pose.orientation.y)
    wrist.append(pose.orientation.z)
    wrist.append(pose.orientation.w)
    return wrist

  
  def getPoseWrist(self):
    """[px, py, pz, ox, oy, oz, ow, fd]"""
    return self.getArmPoseWrist() + self.getHandPose()


  def moveToArmPoseWrist(self, goal_pose):
    """[px, py, pz, ox, oy, oz, ow]"""
    if len(goal_pose) != 7:
      return False
    target = PoseStamped()
    target.header.frame_id = "panda_link0"
    target.pose.position.x = goal_pose[0]
    target.pose.position.y = goal_pose[1]
    target.pose.position.z = goal_pose[2]
    target.pose.orientation.x = goal_pose[3]
    target.pose.orientation.y = goal_pose[4]
    target.pose.orientation.z = goal_pose[5]
    target.pose.orientation.w = goal_pose[6]
    try:
      plan = self.arm.plan(target)
      if plan.joint_trajectory.points:
        self.arm.execute(plan)
        return True
      return False
    except MoveItCommanderException:
      return False


  def moveToPoseWrist(self, goal_pose):
    """[px, py, pz, ox, oy, oz, ow, fd] fd is the distance between the two fingers"""
    if len(goal_pose) != 8:
      return False
    target = PoseStamped()
    target.header.frame_id = "panda_link0"
    target.pose.position.x = goal_pose[0]
    target.pose.position.y = goal_pose[1]
    target.pose.position.z = goal_pose[2]
    target.pose.orientation.x = goal_pose[3]
    target.pose.orientation.y = goal_pose[4]
    target.pose.orientation.z = goal_pose[5]
    target.pose.orientation.w = goal_pose[6]
    gripper = [goal_pose[7]/2.0, goal_pose[7]/2.0]
    try:
      plan = self.arm.plan(target)
      self.hand.plan(gripper)
      if not plan.joint_trajectory.points:
        return False
      self.arm.execute(plan)
      self.hand.go(wait=True)
    except MoveItCommanderException:
      return False
    return True


  # TCP POSE----------------------------------------------------------  
  def getWristFromTCP(self, tcp_pose):
    """Get the world-to-wrist (panda_link8) pose"""
    # Get world -> tcp transform
    world_to_tcp = TransformStamped()
    world_to_tcp.header.frame_id = "panda_link0"
    world_to_tcp.child_frame_id = "tcp"
    world_to_tcp.transform.translation.x = tcp_pose[0]
    world_to_tcp.transform.translation.y = tcp_pose[1]
    world_to_tcp.transform.translation.z = tcp_pose[2]
    world_to_tcp.transform.rotation.x = tcp_pose[3]
    world_to_tcp.transform.rotation.y = tcp_pose[4]
    world_to_tcp.transform.rotation.z = tcp_pose[5]
    world_to_tcp.transform.rotation.w = tcp_pose[6]

    # Get tcp -> wrist transform
    tcp_to_wrist = self.tf_buffer.lookup_transform("tcp", "panda_link8", rospy.Time())

    # Get world -> wrist with transforms composition
    world_to_wrist = TransformStamped()
    world_to_wrist.transform.translation.x = world_to_tcp.transform.translation.x + tcp_to_wrist.transform.translation.x
    world_to_wrist.transform.translation.y = world_to_tcp.transform.translation.y + tcp_to_wrist.transform.translation.y
    world_to_wrist.transform.translation.z = world_to_tcp.transform.translation.z + tcp_to_wrist.transform.translation.z
    q1 = [world_to_tcp.transform.rotation.x, world_to_tcp.transform.rotation.y, world_to_tcp.transform.rotation.z, world_to_tcp.transform.rotation.w]
    q2 = [tcp_to_wrist.transform.rotation.x, tcp_to_wrist.transform.rotation.y, tcp_to_wrist.transform.rotation.z, tcp_to_wrist.transform.rotation.w]
    q3 = quaternion_multiply(q1, q2)
    world_to_wrist.transform.rotation.x = q3[0]
    world_to_wrist.transform.rotation.y = q3[1]
    world_to_wrist.transform.rotation.z = q3[2]
    world_to_wrist.transform.rotation.w = q3[3]

    # Return results
    trans = world_to_wrist.transform.translation
    rot = world_to_wrist.transform.rotation
    return [trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w]
  

  def getArmPoseTCP(self):
    """[px, py, pz, ox, oy, oz, ow] get the world-to-tcp (tool center point) pose"""
    t = self.tf_buffer.lookup_transform("panda_link0", "tcp", rospy.Time())
    trans = t.transform.translation
    rot = t.transform.rotation
    return [trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w]


  def getPoseTCP(self):
    """[px, py, pz, ox, oy, oz, ow, fd] fd is the distance between the two fingers"""
    return self.getArmPoseTCP() + self.getHandPose()
  

  def moveToArmPoseTCP(self, tcp_goal_pose):
    """[px, py, pz, ox, oy, oz, ow]"""
    # print("prima: ", goal_pose)
    wrist_goal_pose = self.getWristFromTCP(tcp_goal_pose)
    # print("dopo: ", goal_pose)
    return self.moveToArmPoseWrist(wrist_goal_pose)


  def moveToPoseTCP(self, tcp_goal_pose):
    """[px, py, pz, ox, oy, oz, ow, fd] fd is the distance between the two fingers"""
    # print("prima: ", goal_pose)    
    wrist_goal_pose = self.getWristFromTCP(tcp_goal_pose[:7])
    wrist_goal_pose.append(tcp_goal_pose[7])
    # print("dopo: ", goal_pose)
    return self.moveToPoseWrist(wrist_goal_pose)


  # INTERFACE POSE----------------------------------------------------
  def getPose(self):
    """[px, py, pz, ox, oy, oz, ow, fd] fd is the distance between the two fingers"""
    return self.getPoseTCP()


  def moveToPose(self, tcp_goal_pose):
    """[px, py, pz, ox, oy, oz, ow, fd, gr]
        fd is the distance between the two fingers
        gr is 1 if the grasp is active, otherwise 0"""
    if self.moveToPoseTCP(tcp_goal_pose[:8]):
      if tcp_goal_pose[8] == 1:
        print("GRASPING...")
      return True
    return False


  def execute_cartesian_path(self, waypoints, eef_step = 0.01, jump_threashould = 0.0):
    # Generate planning
    (plan, fraction) = self.arm.compute_cartesian_path(
                                      waypoints,        # waypoints to follow
                                      eef_step,         # interpolation
                                      jump_threashould) # jump_threshold -> with 0 not check invalid jumps in joint space
    # Execute planning
    self.arm.execute(plan, wait=True)
  


def test1(sys):
  try:
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

    # Inizialize movegroupinterface
    panda = PandaMoveitInterface(delay=1)


    # Perform custom commands
    while True:
      cmd = input("Insert command: ")

      if (cmd == "quit"):
        return
      
      elif (cmd == "print"):
        print("HAND JOINTS:")
        print(panda.getHandJoints())
        print("--------------------\n")
        print("ARM JOINTS: ")
        print(panda.getArmJoints())
        print("--------------------\n")
        print("ALL JOINTS:")
        print(panda.getJoints())
        print("--------------------\n")

        print("HAND POSE:")
        print(panda.getHandPose())
        print("--------------------\n")
        print("ARM WRIST POSE:")
        print(panda.getArmPoseWrist())
        print("--------------------\n")
        print("ARM TCP POSE:")
        print(panda.getArmPoseTCP())
        print("--------------------\n")
        print("ALL WRIST POSE:")
        print(panda.getPoseWrist())
        print("--------------------\n")
        print("ALL TCP POSE:")
        print(panda.getPoseTCP())
        print("--------------------\n")

      

      # JOINTS------------------------------------------------------------
      elif (cmd == "ready"):
        print("Success? ", panda.moveToReady())


      elif (cmd == "hand j1"):
        print("Success? ", panda.moveToHandJoints(0))
      elif (cmd == "hand j2"):
        print("Success? ", panda.moveToHandJoints(0.035))
      elif (cmd == "hand j3"):
        print("Success? ", panda.moveToHandJoints(0.01))
      elif (cmd == "hand j4"):
        # FAIL TEST
        print("Success? ", panda.moveToHandJoints(1.0))
      

      elif (cmd == "arm j1"):
        print("Success? ", panda.moveToArmJoints((0, -pi/4, 0, -pi/2, 0, pi/3, 0)))
      elif (cmd == "arm j2"):
        # FAIL TEST
        print("Success? ", panda.moveToArmJoints((0, 0, 0, 0, 0, 0, 0))) 
      

      elif (cmd == "j1"):
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.0)))
      elif (cmd == "j2"):
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 0.01)))
      elif (cmd == "j3"):
        # FAIL ARM TEST
        print("Success? ", panda.moveToJoints((0, 0, 0, 0, 0, 0, 0, 0.0))) 
      elif (cmd == "j4"):
        # FAIL HAND TEST
        print("Success? ", panda.moveToJoints((-1.92, -0.25, 2.27, -2.65, -2.58, 0.37, 0.15, 1.0))) 


      # HAND POSE---------------------------------------------------------
      elif (cmd == "hand p1"):
        print("Success? ", panda.moveToHandPose(0))
      elif (cmd == "hand p2"):
        print("Success? ", panda.moveToHandPose(0.07))
      elif (cmd == "hand p3"):
        print("Success? ", panda.moveToHandPose(0.01))
      elif (cmd == "hand p4"):
        # FAIL TEST
        print("Success? ", panda.moveToHandPose(1.0))


      # WRIST POSE--------------------------------------------------------
      elif (cmd == "arm wrist p1"):
        print("Success? ", panda.moveToArmPoseWrist((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0)))
      elif (cmd == "arm wrist p2"):
        # FAIL TEST
        print("Success? ", panda.moveToArmPoseWrist((0, 0, 0, 0, 0, 0, 0)))


      elif (cmd == "wrist p1"):
        print("Success? ", panda.moveToPoseWrist((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.06)))
      elif (cmd == "wrist p2"):
        print("Success? ", panda.moveToPoseWrist((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.01)))
      elif (cmd == "wrist p3"):
        print("Success? ", panda.moveToPoseWrist((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.00)))
      elif (cmd == "wrist p4"):
        # STRANGE TEST
        print("Success? ", panda.moveToPoseWrist((0.4, 0.1, 0.4, 0, 0, 0, 0, 0)))
      elif (cmd == "wrist p5"):
        # FAIL ARM TEST
        print("Success? ", panda.moveToPoseWrist((0, 0, 0, 0, 0, 0, 0, 0)))
      elif (cmd == "wrist p6"):
        # FAIL HAND TEST
        print("Success? ", panda.moveToPoseWrist((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0)))


      # TCP POSE----------------------------------------------------------
      elif (cmd == "arm p1"):
        print("Success? ", panda.moveToArmPoseTCP((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0)))
      elif (cmd == "arm p2"):
        # FAIL TEST
        print("Success? ", panda.moveToArmPoseTCP((0, 0, 0, 0, 0, 0, 0)))


      elif (cmd == "p1"):
        print("Success? ", panda.moveToPoseTCP((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.06)))
      elif (cmd == "p2"):
        print("Success? ", panda.moveToPoseTCP((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.01)))
      elif (cmd == "p3"):
        print("Success? ", panda.moveToPoseTCP((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.00)))
      elif (cmd == "p4"):
        # STRANGE TEST
        print("Success? ", panda.moveToPoseTCP((0.4, 0.1, 0.4, 0, 0, 0, 0, 0)))
      elif (cmd == "p5"):
        # FAIL ARM TEST
        print("Success? ", panda.moveToPoseTCP((0, 0, 0, 0, 0, 0, 0, 0)))
      elif (cmd == "p6"):
        # FAIL HAND TEST
        print("Success? ", panda.moveToPoseTCP((0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0)))
      
      else:
        print("Code not valid")

  except rospy.ROSInterruptException:
    print("ROS interrupted")
    return
  except KeyboardInterrupt:
    print("Keboard quit")
    return


def test2(sys):
  try:
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

    # Inizialize movegroupinterface
    panda = PandaMoveitInterface(delay=1)

    # current = panda.getArmPoseWrist()
    current = panda.getArmJoints()

    for t in range(10):
      current[0] += 0.05
      # print(panda.moveToArmPoseWrist(current))
      print(panda.moveToArmJoints(current))
      print("time step: ", t)

    for t in range(10):
      current[0] -= 0.05
      # print(panda.moveToArmPoseWrist(current))
      print(panda.moveToArmJoints(current))
      print("time step: ", t)

  except rospy.ROSInterruptException:
    print("ROS interrupted")
    return



if __name__ == '__main__':
  """TEST"""
  import sys
  test1(sys)
  # test2(sys)