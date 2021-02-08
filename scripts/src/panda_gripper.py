#! /usr/bin/env python2

import rospy
import actionlib

from franka_gripper.msg import HomingAction, MoveAction, GraspAction, StopAction
from franka_gripper.msg import HomingGoal, MoveGoal, GraspGoal, StopGoal



class PandaGripperInterface():
    def __init__(self, startup_homing = False): 
        # Create action clients
        self._client_homing = actionlib.SimpleActionClient('franka_gripper/homing', HomingAction)
        self._client_move = actionlib.SimpleActionClient('franka_gripper/move', MoveAction)
        self._client_grasp = actionlib.SimpleActionClient('franka_gripper/grasp', GraspAction)
        self._client_stop = actionlib.SimpleActionClient('franka_gripper/stop', StopAction)

        # Wait action servers
        self._client_homing.wait_for_server()
        self._client_move.wait_for_server()
        self._client_grasp.wait_for_server()
        self._client_stop.wait_for_server()

        # Other
        self.MIN_WIDTH = 0.00 # [m] closed
        self.MAX_WIDTH = 0.08 # [m] opened
        self.MIN_FORCE = 0.01 # [N]
        self.MAX_FORCE = 50.0 # [N]
        self._timeout = 10.0

        if startup_homing:
            self.homing()

    def _normalize(self, val, min, max):
        if val < min:
            return min
        elif val > max:
            return max
        else:
            return val 


    def safe_force(self, val):
        print("Force Adjusted!")
        self._normalize(val, self.MIN_FORCE, self.MAX_FORCE)


    def safe_width(self, val):
        print("Width Adjusted!")
        self._normalize(val, self.MIN_WIDTH, self.MAX_WIDTH)


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
        force = self.safe_force(force)
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))
    

    def stop(self):
        # Create goal
        goal = StopGoal()
        # Send goal
        self._client_homing.send_goal_and_wait(goal, rospy.Duration.from_sec(self._timeout))



if __name__ == '__main__':
    rospy.init_node('panda_gripper_node')

    print("Create PandaGripperInterface")
    gripper = PandaGripperInterface(startup_homing=False)

    print("PandaGripperInterface.homing()")
    gripper.homing()
    rospy.sleep(1)

    print("PandaGripperInterface.move()")
    gripper.move(width=0.07, speed=0.1)
    rospy.sleep(1)

    print("PandaGripperInterface.grasp()")
    gripper.grasp(width=0.07, epsilon_inner=0.02, epsilon_outer=0.02, speed=0.1, force=10)
    rospy.sleep(1)