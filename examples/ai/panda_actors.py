#!/usr/bin/env python3

# Custom
import sys
sys.path.append("../../scripts/")
from src.utils import quaternion_multiply, transform
from src.colors import print_col, colorize

# Other
from .models import actor
from .arguments import get_args
import torch
import numpy as np
import time



class AiActor():
    def __init__(self, DEBUG_ENABLED=False, MAX_EPISODE_STEPS=50):
        # attributes
        self.debug_enabled = DEBUG_ENABLED
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.phase = 0    # 0=finish, 1=pre-grasp, 2=grasp, 3=post-grasp
        self.timer = 0
        self.phase_change_delay = 1                      # [sec]
        
        self.approach_position_tollerance      = 0.031   # [m]
        self.approach_orientation_tollerance   = 0.005   # [m]
        self.manipulate_position_tollerance    = 0.020   # [m]
        self.manipulate_orientation_tollerance = 0.005   # [m]
        self.retract_position_tollerance       = 0.031   # [m]
        self.retract_orientation_tollerance    = 0.005   # [m]

        # load ai
        self._loadAI()
    

    def _loadAI(self):
        # get arguments
        self.args = get_args()

        # load pre-grasp model [approach]
        model_path_approach = "ai/" + self.args.save_dir + self.args.env_name + '/approach.pt'
        self.o_mean_approach, self.o_std_approach, self.g_mean_approach, self.g_std_approach, model_approach = torch.load(model_path_approach, map_location=lambda storage, loc: storage)
        
        # load grasp model [manipulate]
        model_path_manipulate = "ai/" + self.args.save_dir + self.args.env_name + '/manipulate.pt'
        self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
        
        # load post-grasp model [retract]
        model_path_retract = "ai/" + self.args.save_dir + self.args.env_name + '/retract.pt'
        self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, model_retract = torch.load(model_path_retract, map_location=lambda storage, loc: storage)
        
        # get the environment params
        env_params = {
                        'obs': 25,         # observation['observation'].shape[0]
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
    
                    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def setMaxEpisodeSteps(self, max_steps):
        self.max_episode_steps = max_steps


    def _process_inputs(self, o, g, o_mean, o_std, g_mean, g_std, args):
        o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
        g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return inputs
                
    
    def reset(self, goal_pose, objOnStart_pose, preGrasp_pose):
        # reset attributes
        self.phase = 1
        self.timer = 0

        # get goal pose
        self.goal_pose = goal_pose

        # get object pose on start
        self.objOnStart_pose = objOnStart_pose
        
        # generate pre_grasp pose
        self.preGrasp_pose = preGrasp_pose

        # debug
        self._debugPrint("[ai   ] Goal pose {}".format(goal_pose.tolist()), 'FG_MAGENTA')
        self._debugPrint("[ai   ] ObjectOnStart pose {}".format(objOnStart_pose.tolist()), 'FG_MAGENTA')
        self._debugPrint("[ai   ] PreGrasp pose {}\n".format(preGrasp_pose.tolist()), 'FG_MAGENTA')


    def getAction(self, obs, current_pose, current_gripper):
        action = self._policy(obs, current_pose, current_gripper)
        self.timer += 1

        # debug
        self._debugPrint("[ai   ] Current pose {}".format(current_pose.tolist()), 'FG_MAGENTA')
        self._debugPrint("[ai   ] Current gripper {}".format(current_gripper), 'FG_MAGENTA')
        self._debugPrint("[ai   ] action {}\n".format(action.tolist()), 'FG_MAGENTA')
        self._debugPrint("[ai   ] Obs {}".format(obs), 'FG_MAGENTA')
        return action


    def _policy(self, obs, current_pose, current_gripper):
        # PRE-GRASP
        if self.phase == 1:
            if self.timer <= 20 and \
                (np.linalg.norm(self.preGrasp_pose[:3] - current_pose[:3]) >= self.approach_position_tollerance or \
                 np.linalg.norm(self.preGrasp_pose[3:] - current_pose[3:]) >= self.approach_orientation_tollerance): 
                with torch.no_grad():
                    input_tensor = self._process_inputs(obs, self.preGrasp_pose[:3], self.o_mean_approach, self.o_std_approach, self.g_mean_approach, self.g_std_approach, self.args)
                    pi = self.actor_network_approach(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(current_pose[3:], self.preGrasp_pose[3:])
                grip = [1] # open gripper
                return np.append(np.append(position, orientation), grip) # action
            else:
                self.phase = 2
                self.timer = 0
                time.sleep(self.phase_change_delay)
            
        # GRASP
        if self.phase == 2: 
            if self.timer < self.max_episode_steps and \
                (np.linalg.norm(self.objOnStart_pose[:3] - current_pose[:3]) >= self.manipulate_position_tollerance or \
                 np.linalg.norm(self.objOnStart_pose[3:] - current_pose[3:]) >= self.manipulate_orientation_tollerance):
                with torch.no_grad():
                    input_tensor = self._process_inputs(obs, self.objOnStart_pose[:3], self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, self.args)
                    pi = self.actor_network_manipulate(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(current_pose[3:], self.objOnStart_pose[3:])
                grip = [1] # open gripper
                return np.append(np.append(position, orientation), grip) # action
            else:
                self.phase = 3
                self.timer = 0
                time.sleep(self.phase_change_delay)
   
        # POST-GRASP
        if self.phase == 3:
            if self.timer < self.max_episode_steps and \
                (np.linalg.norm(self.goal_pose[:3] - current_pose[:3]) >= self.retract_position_tollerance or \
                 np.linalg.norm(self.goal_pose[3:] - current_pose[3:]) >= self.retract_orientation_tollerance): 
                with torch.no_grad():
                    input_tensor = self._process_inputs(obs, self.goal_pose[:3], self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, self.args)
                    pi = self.actor_network_retract(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(current_pose[3:], self.goal_pose[3:])
                grip = [-1] # close gripper
                return np.append(np.append(position, orientation), grip) # action
            else:
                self.phase = 0
                time.sleep(self.phase_change_delay)


        # FINISH
        if self.phase == 0:
            return np.array([0., 0., 0.,  0., 0., 0., 1.,  1]) # action


    def getPhase(self):
        return self.phase


    def goalIsAchieved(self):
        return self.phase == 0