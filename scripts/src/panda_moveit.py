#!/usr/bin/env python2

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

# Ros and Moveit
import rospy
from moveit_commander.exception import MoveItCommanderException
import moveit_commander

# Gripper controller
from panda_gripper import PandaGripperInterface, normalize

# TF2
import tf2_ros
from tf.transformations import quaternion_multiply
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

# Other
from math import pi
import time

from utils import transform



class PandaMoveitInterface(object):
    def __init__(self, delay = 0, real_robot = False):
        super(PandaMoveitInterface, self).__init__()
        # arm settings
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")

        # transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 

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
    def getWristFromTCP(self, goal_pose):
        """Get the world-to-wrist (panda_link8) pose"""
        # Get world -> tcp transform
        world_to_tcp = TransformStamped()
        world_to_tcp.header.frame_id = "panda_link0"
        world_to_tcp.child_frame_id = "tcp"
        world_to_tcp.transform.translation.x = goal_pose[0]
        world_to_tcp.transform.translation.y = goal_pose[1]
        world_to_tcp.transform.translation.z = goal_pose[2]
        world_to_tcp.transform.rotation.x = goal_pose[3]
        world_to_tcp.transform.rotation.y = goal_pose[4]
        world_to_tcp.transform.rotation.z = goal_pose[5]
        world_to_tcp.transform.rotation.w = goal_pose[6]

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


    def movePose(self, goal_pose):
        """ [px, py, pz, ox, oy, oz, ow, fd, gr]
            fd is the distance between the two fingers
            gr is 1 if the grasp is active, otherwise 0"""
        print("goal: ", goal_pose)
        if self.moveArmPoseTCP(goal_pose[:7]):
            if goal_pose[8] == 1:
                self.graspGripper(width=goal_pose[7])
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
    try:
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

    except rospy.ROSInterruptException:
        print("ROS interrupted")
        return




def test3(sys):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

        # Inizialize movegroupinterface
        panda = PandaMoveitInterface(delay=1)


        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer) 

        tcp_to_wrist = tf_buffer.lookup_transform("tcp", "panda_link8", rospy.Time())
        print("tcp_to_wrist: {}\n".format(tcp_to_wrist))

        wrist_to_tcp = tf_buffer.lookup_transform("panda_link8", "tcp", rospy.Time())
        print("wrist_to_tcp: {}\n".format(wrist_to_tcp))



        print("----------------------------")
        tcp_to_wrist = [3.85185988877e-34, -1.26750943712e-17, 0.1035,  -0.923879532511, 0.382683432365, 2.34326020266e-17, 5.65713056144e-17]
        print("tcp_to_wrist: {}\n".format(tcp_to_wrist))

        wrist_to_tcp = [0.0, 0.0, 0.1035,  0.923879532511,  -0.382683432365, -2.34326020266e-17, 5.65713056144e-17]
        print("wrist_to_tcp: {}\n".format(wrist_to_tcp))


        print("----------------------------")
        world_to_tcp = [0.3, 0, 0.5,  0, 0, 0, 1]
        print("world_to_tcp: ", world_to_tcp)

        world_to_wirst = transform(world_to_tcp, tcp_to_wrist)
        print("world_to_wirst: ", world_to_wirst)

        world_to_tcp = transform(world_to_wirst, wrist_to_tcp)
        print("world_to_tcp: ", world_to_tcp)

    except rospy.ROSInterruptException:
        print("ROS interrupted")
        return



if __name__ == '__main__':
    """TEST"""
    import sys
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_demo_node', anonymous=True)

    # test_type('joints')
    # test_type('wrist')
    test_type('tcp')
    


  