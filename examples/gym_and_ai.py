#!/usr/bin/env python3

import sys
sys.path.append("../scripts/")

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import gym
import panda_gym
import numpy as np
import time



def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([ x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0 ], dtype=np.float64)


def transform2(pose1, pose2):
    position = pose1[:3] + pose2[:3]
    orientation = quaternion_multiply(pose1[3:], pose2[3:])
    return np.concatenate((position, orientation), axis=None)


def transform3(pose1, pose2, pose3):
    return transform2(transform2(pose1, pose2), pose3)



class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, DEBUG_ENABLED, MODE, HOST, PORT):
        # demo parameters
        self.debug_enabled = DEBUG_ENABLED
        self.mode = MODE           # sim or real
        
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym] trasformation
        self.tolerance = 0.005        # [m]
        self.phase_change_delay = 1   # [sec]
        self.obj_width = 0.04         # [m]
        self.gripper_move_steps = 10  # [step]
        self.gripper_step = 0
        self.phase = 0  # 1=pre-grasp, 2=grasp, 3=close, 4=place

        # initialize
        if MODE == "real":
            self._robot_init(HOST, PORT)
        elif MODE == "sim":
            self._gym_init("PandaPickAndPlace-v0", True)
            print_col("panda -> gym: {}\n".format(self.panda_to_gym.tolist()), 'FG_MAGENTA')
        else:
            print_col("Wrong mode: ignored", "FG_RED")
    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def goalAchieved(self):
        return self.phase == 0
    
    
    def reset(self):
        self.phase = 1  

        if MODE == "real":
            self._robot_reset()
            self._debugPrint("[real ] goal: {}\n".format((self.goal_pose).tolist()), 'FG_MAGENTA')
            self._debugPrint("[panda] start: {}".format((self.current_pose).tolist()), 'FG_WHITE')
        elif MODE == "sim":
            self._gym_reset()
            self._debugPrint("[gym  ] goal: {}".format(self.goal_pose.tolist()), 'FG_MAGENTA')
            self._debugPrint("[panda] goal: {}".format(transform2(self.panda_to_gym, self.goal_pose).tolist()), 'FG_MAGENTA')
            self._debugPrint("[panda] objOnStart: {}".format(transform2(self.panda_to_gym, self.objOnStart_pose).tolist()), 'FG_MAGENTA')
            self._debugPrint("[panda] preGraspGoal: {}\n".format(transform2(self.panda_to_gym, self.preGraspGoal_pose).tolist()), 'FG_MAGENTA')
            
            self._debugPrint("[panda] start: {}".format(transform2(self.panda_to_gym, self.current_pose).tolist()), 'FG_BLUE')
        self._debugPrint("")

        
    def getAction(self):
        # Generate action with policy
        self.policy2()
        self._debugPrint("action: {}".format(self.action.tolist()), 'FG_MAGENTA')
        print_col("action final: {}\n".format(np.append(self.action[:3]*0.05, self.action[3:]).tolist()), 'FG_MAGENTA')
        # target_pose_tmp = transform2(self.current_pose, self.action[:7])
        self.target_pose = transform2(self.current_pose, np.append(self.action[:3]*0.05, self.action[3:7]))
        self.target_gripper = self.obj_width if self.action[7] < 0 else 0.08

        if MODE == "real":
            # self._debugPrint("[real ] Target: {}".format(target_pose_tmp.tolist()), 'FG_WHITE')
            self._debugPrint("[real ] Target final: {}".format(self.target_pose.tolist()), 'FG_WHITE')
        elif MODE == "sim":
            self.env.render()
            # target_pose_tmp = transform2(self.panda_to_gym, target_pose_tmp)
            self.target_pose = transform2(self.panda_to_gym, self.target_pose)
            # self._debugPrint("[panda] Target: {}".format(target_pose_tmp.tolist()), 'FG_BLUE')
            self._debugPrint("[panda] Target final: {}".format(self.target_pose.tolist()), 'FG_BLUE')
        self._debugPrint("")
    
    
    def step(self):
        if MODE == "real":
            self._robot_step()
            self._debugPrint("[real ] goal: {}\n".format((self.goal_pose).tolist()), 'FG_MAGENTA')
            self._debugPrint("[panda] current: {}".format((self.current_pose).tolist()), 'FG_WHITE')
        elif MODE == "sim":
            self._gym_step()            
            self._debugPrint("[panda] current: {}".format(transform2(self.panda_to_gym, self.current_pose).tolist()), 'FG_BLUE')
        self._debugPrint("")


    def __del__(self):
        if MODE == "real":
            self._robot_del()
            print_col("close communication", 'FG_MAGENTA')
        elif MODE == "sim":
            self._gym_del()


    """POLICY GENERATION"""
    def policy(self):
        self.action = np.zeros(8)
        # PRE-GRASP APPROCH
        if self.phase == 1:
            if np.linalg.norm(self.preGraspGoal_pose[:3] - self.current_pose[:3]) >= self.tolerance or \
            np.linalg.norm(self.preGraspGoal_pose[3:] - self.current_pose[3:]) >= self.tolerance:
                self.action[0] = (self.preGraspGoal_pose[0] - self.current_pose[0]) * 6
                self.action[1] = (self.preGraspGoal_pose[1] - self.current_pose[1]) * 6
                self.action[2] = (self.preGraspGoal_pose[2] - self.current_pose[2]) * 6
                self.action[3:7] = quaternion_multiply(self.preGraspGoal_pose[3:], self.current_pose[3:])
                self.action[7] = 1 # open gripper
            else:
                self.phase = 2
                print_col("PRE-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # GRASP APPROCH
        if self.phase == 2: 
            if np.linalg.norm(self.objOnStart_pose[:3] - self.current_pose[:3]) >= self.tolerance or \
            np.linalg.norm(self.objOnStart_pose[3:] - self.current_pose[3:]) >= self.tolerance:
                self.action[0] = (self.objOnStart_pose[0] - self.current_pose[0]) * 6
                self.action[1] = (self.objOnStart_pose[1] - self.current_pose[1]) * 6
                self.action[2] = (self.objOnStart_pose[2] - self.current_pose[2]) * 6
                self.action[3:7] = quaternion_multiply(self.objOnStart_pose[3:], self.current_pose[3:])
                self.action[7] = 1 # open gripper
            else:
                self.phase = 3
                self.gripper_step = 0
                print_col("GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # CLOSE GRIPPER
        if self.phase == 3: 
            if self.gripper_step < self.gripper_move_steps:
                self.action = np.array([0, 0, 0,  0, 0, 0, 1,  -1]) # close gripper
                self.gripper_step += 1
            else:
                self.phase = 4
                self.gripper_step = 0
                print_col("GRIPPER CLOSE: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # PLACE APPROCH
        if self.phase == 4:
            if np.linalg.norm(self.goal_pose[:3] - self.current_pose[:3]) >= self.tolerance or \
            np.linalg.norm(self.goal_pose[3:] - self.current_pose[3:]) >= self.tolerance:
                self.action[0] = (self.goal_pose[0] - self.current_pose[0]) * 6
                self.action[1] = (self.goal_pose[1] - self.current_pose[1]) * 6
                self.action[2] = (self.goal_pose[2] - self.current_pose[2]) * 6
                self.action[3:7] = quaternion_multiply(self.goal_pose[3:], self.current_pose[3:])
                self.action[7] = -1 # close gripper
            else:
                self.phase = 0
                print_col("POST-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)
        
        if self.phase == 0: # limit the number of timesteps in the episode to a fixed duration
            self.action = np.array([0, 0, 0,  0, 0, 0, 1,  -1]) # close gripper
    

    def policy2(self):
        self.action = np.array([0.10, 0, 0,  0, 0, 0, 1,  -1]) # close gripper
        self.gripper_step += 1
        if self.gripper_step > 10:
            self.action = np.array([0, 0, 0,  0, 0, 0, 1,  1]) # open gripper
            self.phase = 0
            time.sleep(self.phase_change_delay)


    """GYM BEHAVIOUR"""
    def _gym_init(self, scenario, render):
        # create gym environment
        self.env = gym.make(scenario, render=render)


    def _gym_reset(self):
        # start to do the demo
        observation = self.env.reset()

        # get object pose on start
        objOnStart_posit = observation["observation"][3:6]         # object_pos
        objOnStart_orien = [0, 0, 0, 1]
        self.objOnStart_pose = np.concatenate((objOnStart_posit, objOnStart_orien), axis=None)

        # get goal pose
        goal_posit = observation["desired_goal"]
        goal_orien = [0, 0, 0, 1]
        self.goal_pose = np.concatenate((goal_posit, goal_orien), axis=None)

        # generate pre_grasp pose
        self.preGraspGoal_pose = self.objOnStart_pose.copy()
        self.preGraspGoal_pose[2] += 0.10  # [m] above the obj

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state

        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)


    def _gym_step(self):
        # put actions into the environment and get observations
        action = self.action[:3].tolist()
        action.append(self.action[7])
        observation, reward, done, info = self.env.step(action)

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state

        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)


    def _gym_del(self):
        # close gym environment
        self.env.close()


    """REAL ROBOT BEHAVIOUR"""
    def _robot_init(self, HOST, PORT):
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection


    def _robot_reset(self):
        start_msg = [0.3, 0.0, 0.6,  0.0, 0.0, 0.0, 1.0,  0.08, 0]
        self.goal_pose = np.array(start_msg[:7])
        self.goal_pose[0] -= 0.1
    
        print_col("[real] Goal: {}".format(self.goal_pose.tolist()), 'FG_MAGENTA')

        self.panda.sendGoalState(start_msg) 
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()
            
        self.current_pose = np.array(current_msg[:7])
        self.current_gripper = current_msg[7] # fingers width

        self._debugPrint("[real] Start: {}\n".format(self.current_pose.tolist() + [self.current_gripper]), 'FG_BLUE')
        

    def _robot_step(self):
        # Process grasp information
        grasp = 1 if self.action[7] < 0 else 0
            
        # Send goal
        self.panda.sendGoalState(self.target_pose.tolist() + [self.target_gripper, grasp]) # gripper_open

        # Get current pose and gripper state of Panda
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()
            
        self.current_pose = np.array(current_msg[:7])
        self.current_gripper = current_msg[7] # fingers width

        self._debugPrint("[real] Current: {}".format(self.current_pose.tolist() + [self.current_gripper]), 'FG_BLUE')
        self._debugPrint("")


    def _robot_del(self):
        print_col("close communication", 'FG_MAGENTA')
        self.panda.sendClose()



def main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENABLED, MODE, HOST, PORT):
    # initialize Actor
    my_actor = PandaActor(DEBUG_ENABLED, MODE, HOST, PORT)

    for episode in range(NUM_EPISODES):
        # reset actor and get first observations
        my_actor.reset()

        goal_achived = False
        for time_step in range(LEN_EPISODE):
            if DEBUG_ENABLED:
                print_col("[Step {:>3}]------------------------------------------------".format(time_step), 'FG_GREEN')
            
            # generate new action from observations
            my_actor.getAction()

            # perform a step and get new observations
            my_actor.step()

            # check goal
            if my_actor.goalAchieved():
                goal_achived = True
                break

        print_col("Episode {} finish".format(episode), 'FG_GREEN')
        if goal_achived:
            print_col("Goal achived in {} step".format(time_step), 'FG_GREEN_BRIGHT')
        else:
            print_col("Goal not achived in {} step".format(time_step), 'FG_RED_BRIGHT')

    print_col("All Episodes finish", 'FG_GREEN')



if __name__ == "__main__":
    # PARAMETERS
    HOST = "127.0.0.1"
    PORT = 2000
    NUM_EPISODES = 1
    LEN_EPISODE = 100
    DEBUG_ENABLED = False
    MODE = "sim"


    if (len(sys.argv) > 1):
        if sys.argv[1] == 'sim' or sys.argv[1] == "real":
            MODE = sys.argv[1]

    if len(sys.argv) > 2:
        if sys.argv[2] == 'debug':
            DEBUG_ENABLED = True

    if len(sys.argv) > 3:
        LEN_EPISODE = int(sys.argv[3])
            

    main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENABLED, MODE, HOST, PORT)