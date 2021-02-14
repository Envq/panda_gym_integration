#!/usr/bin/env python3

import sys
sys.path.append("../scripts/")

# Panda Interface
from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import numpy as np
import time

# Panda-gym and AI
import gym
import panda_gym
import torch
from rl_modules.models import actor
from rl_modules.arguments import get_args



def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([ x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0 ], dtype=np.float64)


def transform(pose1, pose2):
    position = pose1[:3] + pose2[:3]
    orientation = quaternion_multiply(pose1[3:], pose2[3:])
    return np.concatenate((position, orientation), axis=None)


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, DEBUG_ENABLED, MODE, HOST, PORT):
        # demo parameters
        self.debug_enabled = DEBUG_ENABLED
        self.mode = MODE              # sim or real
        
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
            raise ValueError(colorize("Wrong mode: ignored", "FG_RED"))

        # initialize ai
        self._loadAI()

    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def goalAchieved(self):
        return self.phase == 0
    
    
    def reset(self):
        # Reset phase
        self.phase = 1  

        # Generate goal and get start pose
        if MODE == "sim":
            self._gym_reset()
            objOnStart_pose = transform(self.panda_to_gym, self.objOnStart_pose)
            goal_pose  = transform(self.panda_to_gym, self.goal_pose)
            current_pose = transform(self.panda_to_gym, self.current_pose)
            current_gripper = self.current_gripper.copy()

        elif MODE == "real":
            self._robot_reset()
            objOnStart_pose = self.objOnStart_pose.copy()
            goal_pose  = self.goal_pose.copy()
            current_pose = self.current_pose.copy()
            current_gripper = self.current_gripper.copy()
        
        # generate pre_grasp pose
        self.preGrasp_pose = self.objOnStart_pose.copy()
        if self.current_pose[0] > self.objOnStart_pose[0]:
            self.preGrasp_pose[0] += 0.003
        else:
            self.preGrasp_pose[0] -= 0.003

        if self.current_pose[1] > self.objOnStart_pose[1]:
            self.preGrasp_pose[1] += 0.002
        else:
            self.preGrasp_pose[1] -= 0.002
        self.preGrasp_pose[2] += 0.031  # [m] above the obj

        # Debug
        self._debugPrint("objOnStart_pose: {}".format(objOnStart_pose.tolist()), 'FG_MAGENTA')
        self._debugPrint("goal_pose: {}".format(goal_pose.tolist()), 'FG_MAGENTA')
        self._debugPrint("preGrasp_pose: {}\n".format(transform(self.panda_to_gym, self.preGrasp_pose).tolist()), 'FG_MAGENTA')

        self._debugPrint("start_gripper: {}".format(current_gripper.tolist()), 'FG_WHITE')
        self._debugPrint("start_pose: {}\n".format(current_pose.tolist()), 'FG_WHITE')

        
    def getAction(self, time_step):
        # Generate action with policy
        self._policyAI(time_step)
        action = self.action.copy()
        # action[:3] *= 0.05  # Correct with panda-gym (limit maximum change in position)

        if MODE == "sim":
            self.env.render()
            self.target_pose = transform(transform(self.panda_to_gym, self.current_pose), action[:7])
            self.target_gripper = action[7]

        elif MODE == "real":
            self.target_pose = transform(self.current_pose, action[:7])
            self.target_gripper = self.obj_width if action[7] < 0 else 0.08

        # Debug
        self._debugPrint("action: {}".format(self.action.tolist()), 'FG_MAGENTA')
        print_col("action final: {}\n".format(action.tolist()), 'FG_MAGENTA')
        
        self._debugPrint("target_pose with correction: {}\n".format(self.target_pose.tolist()), 'FG_WHITE')
    
    
    def step(self):
        # Generate goal and get current pose
        if MODE == "sim":
            self._gym_step()
            current_pose = transform(self.panda_to_gym, self.current_pose)
            current_gripper = self.current_gripper.copy()
            obj_pose = transform(self.panda_to_gym, self.obj_pose)

        elif MODE == "real":
            self._robot_step()
            current_pose = self.current_pose.copy()
            current_gripper = self.current_gripper.copy()
            obj_pose = self.obj_pose.copy()

        # Debug
        self._debugPrint("current_gripper: {}".format(current_gripper.tolist()), 'FG_WHITE')
        self._debugPrint("current_pose: {}\n".format(current_pose.tolist()), 'FG_WHITE')
        self._debugPrint("obj_pose: {}\n".format(obj_pose.tolist()), 'FG_WHITE')


    def __del__(self):
        if MODE == "real":
            self._robot_del()
            print_col("close communication", 'FG_MAGENTA')
        elif MODE == "sim":
            self._gym_del()


    """AI BEHAVIOUR"""
    def _loadAI(self):
        # get arguments
        self.args = get_args()

        # load pre-grasp model [approach]
        model_path_approach  = self.args.save_dir + self.args.env_name + '/approach.pt'
        self.o_mean_approach , self.o_std_approach , self.g_mean_approach , self.g_std_approach , model_approach = torch.load(model_path_approach , map_location=lambda storage, loc: storage)
        
        # load grasp model [manipulate]
        model_path_manipulate = self.args.save_dir + self.args.env_name + '/manipulate.pt'
        self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
        
        # load place model [place]
        model_path_retract = self.args.save_dir + self.args.env_name + '/retract.pt'
        self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, model_retract = torch.load (model_path_retract, map_location=lambda storage, loc: storage)
        
        # get the environment params
        env_params = {'obs': 25,         # observation['observation'].shape[0]
                      'goal': 3,         # observation['desired_goal'].shape[0]
                      'action': 4,       # self.env.action_space.shape[0]
                      'action_max':1.0,  # self.env.action_space.high[0]
                     }

        # create the actor network
        self.actor_network_approach = actor(env_params)
        self.actor_network_approach.load_state_dict(model_approach)
        self.actor_network_approach.eval()
        self.actor_network_manipulate = actor(env_params)
        self.actor_network_manipulate.load_state_dict(model_manipulate)
        self.actor_network_manipulate.eval()
        self.actor_network_retract = actor(env_params)
        self.actor_network_retract.load_state_dict(model_retract)
        self.actor_network_retract.eval()


    """POLICY GENERATION and AI"""
    def _policyAI(self, time_step):
        # PRE-GRASP
        if self.phase == 1:
            if np.linalg.norm(self.current_pose[:3] - self.preGrasp_pose[:3]) >= 0.031 or \
               np.linalg.norm(self.current_pose[3:] - self.preGrasp_pose[3:]) >= self.tolerance:
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.preGrasp_pose[:3], self.o_mean_approach, self.o_std_approach, self.g_mean_approach, self.g_std_approach, self.args)
                    pi = self.actor_network_approach(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(self.current_pose[3:], self.preGrasp_pose[3:])
                grip = [1] # open gripper
                self.action = np.append(np.append(position, orientation), grip)
            else:
                self.phase = 2
                print_col("PRE-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # GRASP
        if self.phase == 2: 
            if np.linalg.norm(self.current_pose[:3] - self.obj_pose[:3]) >= 0.015 or \
               np.linalg.norm(self.current_pose[3:] - self.obj_pose[3:]) >= self.tolerance:
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.obj_pose[:3], self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, self.args)
                    pi = self.actor_network_manipulate(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(self.current_pose[3:], self.preGrasp_pose[3:])
                grip = [1] # open gripper
                self.action = np.append(np.append(position, orientation), grip)
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

        # PLACE
        if self.phase == 4:
            if np.linalg.norm(self.goal_pose[:3] - self.obj_pose[:3]) >= 0.010 or \
               np.linalg.norm(self.goal_pose[3:] - self.obj_pose[3:]) >= self.tolerance:
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.goal_pose[:3], self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, self.args)
                    pi = self.actor_network_retract(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(self.current_pose[3:], self.preGrasp_pose[3:])
                grip = [-1] # close gripper
                self.action = np.append(np.append(position, orientation), grip)
            else:
                self.phase = 0
                print_col("POST-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)
        
        if self.phase == 0: # limit the number of timesteps in the episode to a fixed duration
            self.action = np.array([0, 0, 0,  0, 0, 0, 1,  -1]) # close gripper
    

    def _policyInterpolation(self):
        self.action = np.zeros(8)
        # PRE-GRASP APPROCH
        if self.phase == 1:
            if np.linalg.norm(self.preGrasp_pose[:3] - self.current_pose[:3]) >= self.tolerance or \
            np.linalg.norm(self.preGrasp_pose[3:] - self.current_pose[3:]) >= self.tolerance:
                self.action[0] = (self.preGrasp_pose[0] - self.current_pose[0]) * 6
                self.action[1] = (self.preGrasp_pose[1] - self.current_pose[1]) * 6
                self.action[2] = (self.preGrasp_pose[2] - self.current_pose[2]) * 6
                self.action[3:7] = quaternion_multiply(self.preGrasp_pose[3:], self.current_pose[3:])
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
    

    def _policySimple(self):
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
        # reset environment and get first observation
        observation = self.env.reset()
        self.obs = observation["observation"]

        # get object pose on start
        objOnStart_posit = observation["observation"][3:6]         # object_pos
        objOnStart_orien = [0, 0, 0, 1]
        self.objOnStart_pose = np.concatenate((objOnStart_posit, objOnStart_orien), axis=None)

        # get goal pose
        goal_posit = observation["desired_goal"]
        goal_orien = [0, 0, 0, 1]
        self.goal_pose = np.concatenate((goal_posit, goal_orien), axis=None)

        # get current tcp pose (on gym frame)
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)

        # get current fingers width
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state


    def _gym_step(self):
        # put actions into the environment and get observations
        position = self.action[:3].tolist()
        grip = [self.action[7]]
        observation, reward, done, info = self.env.step(position + grip)

        # observation
        self.obs = observation["observation"]

        # get object pose on start
        obj_posit = observation["observation"][3:6]                # object_pos
        obj_orien = [0, 0, 0, 1]
        self.obj_pose = np.concatenate((obj_posit, obj_orien), axis=None)

        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state


    def _gym_del(self):
        # close gym environment
        self.env.close()


    """REAL ROBOT BEHAVIOUR"""
    def _robot_init(self, HOST, PORT):
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection


    def _robot_reset(self):
        # get object pose on start
        self.objOnStart_pose = np.array([0.7019080739083265, -0.11301889621397332, 0.125, 0.0, 0.0, 0.0, 1.0])
        
        # get goal pose
        self.goal_pose = np.array([0.738619682797228, 0.04043141396766836, 0.5272451383552441, 0.0, 0.0, 0.0, 1.0])
        
        # start msg
        start_msg = [0.3, 0.0, 0.6,  0.0, 0.0, 0.0, 1.0,  0.08, 0]           

        # send start state msg
        self.panda.sendGoalState(start_msg) 

        # get current state msg
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()

        # get current tcp pose (on panda_base frame)            
        self.current_pose = np.array(current_msg[:7])  # grip_pos

        # get current fingers width
        self.current_gripper = current_msg[7]          # gripper_state
        

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
            my_actor.getAction(time_step)

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
    NUM_EPISODES = 3
    LEN_EPISODE = 100
    DEBUG_ENABLED = False
    MODE = "sim"

    main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENABLED, MODE, HOST, PORT)