#! /usr/bin/env python2

import rospy
import actionlib

from franka_gripper.msg import HomingAction, MoveAction, GraspAction, StopAction
from franka_gripper.msg import HomingGoal, MoveGoal, GraspGoal, StopGoal



class PandaGripper():
    def __init__(self, startup_homing = False): 
        # Create action clients
        self._client_homing = actionlib.SimpleActionClient('franka_gripper/homing', HomingAction)
        self._client_move_ = actionlib.SimpleActionClient('franka_gripper/move', MoveAction)
        self._client_grasp = actionlib.SimpleActionClient('franka_gripper/grasp', GraspAction)
        self._client_stop = actionlib.SimpleActionClient('franka_gripper/stop', StopAction)

        # Wait action servers
        self._client_homing.wait_for_server()
        self._client_move.wait_for_server()
        self._client_grasp.wait_for_server()

        # Other
        self.GRIPPER_OPEN = 0.08  # m
        self.GRIPPER_CLOSE = 0.00 # m
        self._timeout = 10.0

        if startup_homing:
            self.homing()


    def _safe_width(self, val):
        if val > self.GRIPPER_OPEN:
            return self.GRIPPER_OPEN
        elif val < self.GRIPPER_CLOSE:
            return self.GRIPPER_CLOSE
        else:
            return val 


    def setTimeout(self):
        self._timeout = 10.0
    

    def homing(self):
        # Create goal
        goal = HomingGoal()
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))
    

    def move(self, width, speed):
        # Create goal
        goal = MoveGoal()
        goal.width = self._safe_width(width)
        goal.speed = speed
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))
    

    def grasp(self, width, epsilon_inner, epsilon_outer, speed, force):
        # Create goal
        goal = GraspGoal()
        goal.width = self._safe_width(width)
        epsilon_inner = epsilon_inner
        epsilon_outer = epsilon_outer
        speed = speed
        force = force
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))
    

    def stop(self):
        # Create goal
        goal = StopGoal()
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))



if __name__ == '__main__':
    rospy.init_node('panda_gripper_node')

    print("Create PandaGripper")
    gripper = PandaGripper(startup_homing=False)

    print("PandaGripper.hominge()")
    gripper.homing()

    print("PandaGripper.move()")
    gripper.move(width=0.07, speed=1.0)