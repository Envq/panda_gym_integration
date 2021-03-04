#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

# Ros and Moveit
import rospy
from moveit_commander.exception import MoveItCommanderException
import moveit_commander
from geometry_msgs.msg import PoseStamped

# Custom
from panda_gripper import PandaGripperInterface, normalize
from utils import quaternion_equals, transform, transform_inverse

# Other
from math import pi
import time



class PandaMoveitInterface(object):
    def __init__(self, delay = 0, real_robot = False):
        super(PandaMoveitInterface, self).__init__()
        # arm settings
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")

        # transformations
        self.wrist_to_tcp = [0.0, 0.0, 0.1035, 0.923879533, -0.382683432, 0.0, 0.0]
        self.tcp_to_wrist = transform_inverse(self.wrist_to_tcp).tolist()

        # gripper settings
        self.real_robot = real_robot
        self.current_gripper_width = -1  # use homing for override it
        self.gripper_open_value = PandaGripperInterface.OPEN_VALUE
        if self.real_robot:
            # enable real hand
            self.gripper = PandaGripperInterface(startup_homing=False)
        else:
            # enable fake hand
            self.hand = moveit_commander.MoveGroupCommander("hand")

        # wait for correct loading
        time.sleep(delay) 
      

    # JOINTS------------------------------------------------------------
    def getArmJoints(self):
        """[j0, j1, j2, j3, j4, j5, j6]"""
        return self.arm.get_current_joint_values()


    def moveArmJoints(self, goal_joints):
        """[j0, j1, j2, j3, j4, j5, j6]"""
        if len(goal_joints) != 7:
            return False
        try:
            self.arm.plan(goal_joints)
            self.arm.go(wait=True)
        except MoveItCommanderException:
            return False
        return True
    

    def moveReady(self):
        return self.moveArmJoints((0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi))


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


    def moveArmPoseWrist(self, goal_pose):
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

 
    # TCP POSE----------------------------------------------------------
    def getWristFromTCP(self, world_to_tcp):
        """Get the world-to-wrist (panda_link8) pose"""
        return transform(world_to_tcp, self.tcp_to_wrist).tolist()


    def getArmPoseTCP(self):
        """[px, py, pz, ox, oy, oz, ow] get the world-to-tcp (tool center point) pose"""
        world_to_wrist = self.getArmPoseWrist()
        return transform(world_to_wrist, self.wrist_to_tcp).tolist()

  
    def moveArmPoseTCP(self, goal_pose):
        """[px, py, pz, ox, oy, oz, ow]"""
        wrist_goal_pose = self.getWristFromTCP(goal_pose)
        return self.moveArmPoseWrist(wrist_goal_pose)


    # GRIPPER AND HAND---------------------------------------------------------
    def _moveHand(self, width):
        """[width] is the distance between fingers (min=0.0, max=0.08)"""
        width = normalize(width, 0.0, self.gripper_open_value)
        try:
            self.hand.plan([width/2.0, width/2.0])
            self.hand.go(wait=True)
        except MoveItCommanderException:
            return False
        return True
      
    
    def getGripper(self):
        return self.current_gripper_width


    def homingGripper(self):
        if self.real_robot:
            self.gripper.homing()
        else:
            self._moveHand(self.gripper_open_value)
        self.current_gripper_width = self.gripper_open_value
      

    def moveGripper(self, width, speed=0.5):
        if self.real_robot:
            self.gripper.move(self, width, speed)
        else:
            self._moveHand(width)
        self.current_gripper_width = width
      

    def graspGripper(self, width, speed=0.5, force=10, epsilon_inner=0.02, epsilon_outer=0.02):
        if self.real_robot:
            self.gripper.grasp(self, width, epsilon_inner, epsilon_outer, speed, force)
        else:
            self._moveHand(width)
            print("GRASPING...")
        self.current_gripper_width = width
      
      
    def stopGripper(self):
        if self.real_robot:
            self.gripper.stop()
        else:
            print("ERROR NOT IMPLEMENTED YET")


    # INTERFACE POSE----------------------------------------------------
    def getPose(self):
        """[px, py, pz, ox, oy, oz, ow, fd] fd is the distance between the two fingers"""
        return self.getArmPoseTCP() + [self.getGripper()]


    def movePose(self, goal_pose, grasp_option=None):
        """ [px, py, pz, ox, oy, oz, ow, fd, gr]
            fd is the distance between the two fingers
            gr is 1 if the grasp is active, otherwise 0"""
        if self.moveArmPoseTCP(goal_pose[:7]):
            if goal_pose[8] == 1:
                if grasp_option == None:
                    self.graspGripper(width=goal_pose[7])
                else:
                    self.graspGripper(\
                        width=goal_pose[7], \
                        speed=grasp_option['speed'], \
                        force=grasp_option['force'], \
                        epsilon_inner=grasp_option['epsilon_inner'], \
                        epsilon_outer=grasp_option['epsilon_outer'])     
            else:
                self.moveGripper(width=goal_pose[7])
            return True
        return False


    def execute_cartesian_path(self, waypoints, eef_step = 0.01, jump_threashould = 0.0):
        """Execute cartesian path"""
        # Generate planning
        (plan, fraction) = self.arm.compute_cartesian_path(
                                            waypoints,        # waypoints to follow
                                            eef_step,         # interpolation
                                            jump_threashould) # jump_threshold -> with 0 not check invalid jumps in joint space
        # Execute planning
        self.arm.execute(plan, wait=True)



def test_type(type):
    panda = PandaMoveitInterface(delay=1)

    if (type == 'joints'):
        current = panda.getArmJoints()
    elif (type == 'wrist'):
        current = panda.getArmPoseWrist()
    elif (type == 'tcp'):
        current = panda.getArmPoseTCP()
    else:
        print("error type")
        return
    print("current:", current)

    for t in range(20):
        if t < 10:
            current[0] += 0.025
        else:
            current[0] -= 0.025

        if (type == 'joints'):
            success = panda.moveArmJoints(current)
        elif (type == 'wrist'):
            success = panda.moveArmPoseWrist(current)
        elif (type == 'tcp'):
            success = panda.moveArmPoseTCP(current)

        print("[{:>2} step]: {}".format(t, success))


def test_tf():
    import tf2_ros

    # Inizialize movegroupinterface
    panda = PandaMoveitInterface()

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    time.sleep(2)

    print("WITH TF: ")
    wrist_to_tcp_tf = tf_buffer.lookup_transform("panda_link8", "tcp", rospy.Time())
    wrist_to_tcp = list()
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.translation.x)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.translation.y)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.translation.z)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.rotation.x)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.rotation.y)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.rotation.z)
    wrist_to_tcp.append(wrist_to_tcp_tf.transform.rotation.w)
    print("wrist_to_tcp:     {}".format(wrist_to_tcp))

    tcp_to_wrist_tf = tf_buffer.lookup_transform("tcp", "panda_link8", rospy.Time())
    tcp_to_wrist = list()
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.translation.x)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.translation.y)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.translation.z)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.rotation.x)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.rotation.y)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.rotation.z)
    tcp_to_wrist.append(tcp_to_wrist_tf.transform.rotation.w)
    print("tcp_to_wrist:     {}\n".format(tcp_to_wrist))

    print("WITH UTILS")
    wrist_to_tcp_my = [0.0, 0.0, 0.1035, 0.923879533, -0.382683432, 0.0, 0.0]
    print("wrist_to_tcp:     {}".format(wrist_to_tcp_my))
    
    tcp_to_wrist_my  = transform_inverse(wrist_to_tcp_my)
    print("tcp_to_wrist:     {}".format(tcp_to_wrist_my.tolist()))
    
    wrist_to_tcp_my2  = transform_inverse(tcp_to_wrist_my)
    print("wrist_to_tcp_my2: {}\n".format(wrist_to_tcp_my2.tolist()))

    print("Check TF-MY: ({}, {})".format( \
        quaternion_equals(wrist_to_tcp[3:], wrist_to_tcp_my[3:]), \
        quaternion_equals(tcp_to_wrist[3:], tcp_to_wrist_my[3:])))
    
    print("Check My inv: {}\n".format( \
        quaternion_equals(wrist_to_tcp_my[3:], wrist_to_tcp_my2[3:])))

    
    print("Self.wrist_to_tcp: {}".format(panda.wrist_to_tcp))
    print("Self.tcp_to_wrist: {}".format(panda.tcp_to_wrist))



    print("----------------------------")

    world_to_tcp = [0.3, 0, 0.5,  0, 0, 0, 1]
    print("world_to_tcp:      {}".format(world_to_tcp))

    world_to_wirst_my = transform(world_to_tcp, tcp_to_wrist_my)
    print("world_to_wirst_my: {}".format(world_to_wirst_my.tolist()))

    world_to_wirst_tf = panda.getWristFromTCP(world_to_tcp)
    print("world_to_wirst_tf: {}".format(world_to_wirst_tf))

    print("Check TF-MY:  {}\n".format( \
        quaternion_equals(world_to_wirst_my[3:], world_to_wirst_tf[3:])))

    world_to_tcp_my = transform(world_to_wirst_my, wrist_to_tcp_my)
    print("world_to_tcp_my:   {}".format(world_to_tcp_my.tolist()))
    
    print("Check My inv: {}\n".format( \
        quaternion_equals(world_to_tcp[3:], world_to_tcp_my[3:])))


def test_simple():
    # Inizialize movegroupinterface
    panda = PandaMoveitInterface(delay=1)

    pose = [0.3, 0.0, 0.3,  0, 0, 0, 1]
    panda.moveArmPoseTCP(pose)



if __name__ == '__main__':
    """TEST"""
    import sys
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

    try:
        # test_type('joints')
        # test_type('wrist')
        # test_type('tcp')
        # test_tf()
        test_simple()

    except rospy.ROSInterruptException:
        print("ROS interrupted")
      