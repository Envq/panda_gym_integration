#!/usr/bin/env python3

# Panda Interface
import sys
sys.path.append("../scripts/")
from src.utils import quaternion_multiply, transform
from src.colors import print_col, colorize

# Panda-gym and AI
import gym
import panda_gym
import torch
from ai.models import actor
from ai.arguments import get_args

# Other
import numpy as np
import time
from datetime import datetime
import os



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
    def __init__(self, DEBUG_ENABLED):
        # demo parameters
        self.debug_enabled = DEBUG_ENABLED

        # demo attributes
        self.timeStep = 0
        self.phase = 0                                                      # 0=finish, 1=pre-grasp, 2=grasp, 3=place
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym] trasformation
        self.obj_width = 0.04             # [m]
        self.phase_change_delay = 1       # [sec]
        self.last_phase = self.phase

        # load ai
        self._loadAI()

        # create gym environment
        self.env = gym.make(self.args.env_name, render=True)
            
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def _loadAI(self):
        # get arguments
        self.args = get_args()

        # load pre-grasp model [approach]
        model_path_approach = "ai/" + self.args.save_dir + self.args.env_name + '/approach.pt'
        self.o_mean_approach, self.o_std_approach, self.g_mean_approach, self.g_std_approach, model_approach = torch.load(model_path_approach, map_location=lambda storage, loc: storage)
        
        # load grasp model [manipulate]
        model_path_manipulate = "ai/" + self.args.save_dir + self.args.env_name + '/manipulate.pt'
        self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
        
        # load place model [place]
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


    def reset(self):
        # reset attributes
        self.phase = 1
        self.last_phase = self.phase
        self.timeStep = 0

        # reset environment
        observation = self.env.reset()

        # get goal pose
        goal_posit = observation["desired_goal"]
        goal_orien = [0, 0, 0, 1]
        self.goal_pose = np.concatenate((goal_posit, goal_orien), axis=None)

        # get object pose on start
        objOnStart_posit = observation["observation"][3:6]         # object_pos
        objOnStart_orien = [0, 0, 0, 1]
        self.objOnStart_pose = np.concatenate((objOnStart_posit, objOnStart_orien), axis=None)
        
        # generate pre_grasp pose
        self.preGrasp_pose = self.objOnStart_pose.copy()
        self.preGrasp_pose[2] += 0.031  # [m] above the obj

        # get observation
        self._gym_obs(observation)

        # return start behaviour
        return ([0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916,  \
                -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569], \
                0.08) # open gripper

    
    def _gym_obs(self, observation):
        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state

        # generate obs for AI
        self.obs = np.zeros(25)
        self.obs[:3]    = self.current_pose[:3]     # grip_pos
        self.obs[3:6]   = self.objOnStart_pose[:3]  # object_pos
        self.obs[6:9]   = 0                         # object_rel_pos
        self.obs[9:11]  = [finger0, finger1]        # gripper_state
        self.obs[11:14] = 0                         # object_rot
        self.obs[14:17] = 0                         # object_velp
        self.obs[17:20] = 0                         # object_velr
        self.obs[20:23] = 0                         # grip_velp
        self.obs[23:25] = 0                         # gripper_vel

        # adjust object_pos info for retract-ai (It use object_pos as current_pose info)
        if self.phase == 3:
            self.obs[3:6] = self.current_pose[:3]   # object_pos


    def getTargetInfo(self):
        self.env.render()

        # get action
        self._policy()
        action = self.action.copy()
        action[:3] *= 0.05  # Correct with panda-gym (limit maximum change in position)

        # generate target pose
        current_pose = transform(self.panda_to_gym, self.current_pose)
        target_pose = transform(current_pose, action[:7])

        self._debugPrint("Action: {}".format(action.tolist()), 'FG_WHITE')
        self._debugPrint("Current pose: {}".format(current_pose.tolist()), 'FG_WHITE')
        self._debugPrint("Target pose: {}\n".format(target_pose.tolist()), 'FG_WHITE')

        # generate gripper state
        if self.phase == 3 and self.last_phase == 2:    # close gripper after grasp phase
            gripper_state = self.obj_width

        elif self.phase == 0 and self.last_phase == 3:  # open gripper after place phase
            gripper_state = 0.08

        else:
            gripper_state = 0                           # no operation

        self.last_phase = self.phase                    # update last_phase

        return (target_pose.tolist(), gripper_state)


    def _policy(self):
        # PRE-GRASP
        if self.phase == 1:
            if self.timeStep <= 20 and \
                (np.linalg.norm(self.preGrasp_pose[:3] - self.current_pose[:3]) >= 0.031 or \
                 np.linalg.norm(self.preGrasp_pose[3:] - self.current_pose[3:]) >= 0.005): 
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
                self.timeStep = 0
                time.sleep(self.phase_change_delay)
                self._debugPrint("PRE-GRASP: successful", 'FG_YELLOW_BRIGHT')
            
        # GRASP
        if self.phase == 2: 
            if self.timeStep < self.env._max_episode_steps and \
                (np.linalg.norm(self.objOnStart_pose[:3] - self.current_pose[:3]) >= 0.015 or \
                 np.linalg.norm(self.objOnStart_pose[3:] - self.current_pose[3:]) >= 0.005):
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.objOnStart_pose[:3], self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, self.args)
                    pi = self.actor_network_manipulate(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(self.current_pose[3:], self.objOnStart_pose[3:])
                grip = [1] # open gripper
                self.action = np.append(np.append(position, orientation), grip)
            else:
                self.phase = 3
                self.timeStep = 0
                time.sleep(self.phase_change_delay)
                self._debugPrint("GRASP: successful", 'FG_YELLOW_BRIGHT')
   
        # PLACE
        if self.phase == 3:
            if self.timeStep < self.env._max_episode_steps and \
                (np.linalg.norm(self.goal_pose[:3] - self.current_pose[:3]) >= 0.031 or \
                 np.linalg.norm(self.goal_pose[3:] - self.current_pose[3:]) >= 0.005): 
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.goal_pose[:3], self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, self.args)
                    pi = self.actor_network_retract(input_tensor)
                    action = pi.detach().cpu().numpy().squeeze()
                    position = action[:3]
                orientation = quaternion_multiply(self.current_pose[3:], self.goal_pose[3:])
                grip = [-1] # close gripper
                self.action = np.append(np.append(position, orientation), grip)
            else:
                self.phase = 0
                time.sleep(self.phase_change_delay)
                self._debugPrint("POST-GRASP: successful", 'FG_YELLOW_BRIGHT')

        # FINISH
        if self.phase == 0:
            self.action = np.array([0., 0., 0.,  0., 0., 0., 1.,  1]) # open gripper

        
    def step(self):
        pos = self.action[:3].tolist()
        grip = [self.action[7]]
        observation, reward, done, self.info = self.env.step(pos + grip)
        self._gym_obs(observation)
        self.timeStep += 1


    def checkGoal(self):
        return (self.phase == 0, self.info['is_success'])


    def __del__(self):
        # close gym environment
        self.env.close()



def main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, FILE_PATH, DEBUG_ENABLED):
    # for writing
    trajectory = list()

    # initialize Actor
    my_actor = PandaActor(DEBUG_ENABLED)

    # statistics
    results = {
        'successes': 0,
        'steps': list()
    }

    # start world
    for episode in range(NUM_EPISODES):
        # reset actory
        (target_pose, gripper_state) = my_actor.reset()

        # reset trajectory
        trajectory.clear()

        # add start pose
        trajectory.append(gripper_state)
        trajectory.append(target_pose)
        
        # start episode
        for time_step in range(LEN_EPISODE):
            # generate a new action from observations and create a target pose with it
            (target_pose, gripper_state)  = my_actor.getTargetInfo()

            # add target to trajectory and gripper state
            if gripper_state != 0:
                trajectory.append(gripper_state)
            trajectory.append(target_pose)

            # perform a step and get new observations
            my_actor.step()

            # check goal
            (exit, success) = my_actor.checkGoal()

            # check the output condition
            if exit:
                if success:
                    results['successes'] += 1
                    print_col("[Episode {}] Goal achived in {} steps".format(episode, time_step + 1), 'FG_GREEN_BRIGHT')
                else:
                    print_col("[Episode {}] Goal not achived in {} steps".format(episode, time_step + 1), 'FG_RED_BRIGHT')
                results['steps'].append(time_step + 1)
                break
        
        # write to file
        if WRITE_ENABLE and input("Write to file? [y/n] ") == 'y':
            with open(FILE_PATH, 'w') as file_writer:
                for info in trajectory:
                    # target pose case
                    if type(info) is list:
                        for e in info:
                            file_writer.write("{:>25}  ".format(e))
                        file_writer.write("\n")

                    # gripper state change case
                    else:
                        file_writer.write("{}\n".format(info))
    

    # Generate final statistics
    print("-----------------------------------")
    print_col("All Episodes finish", 'FG_GREEN')

    successes = results['successes']
    fails = NUM_EPISODES - successes
    print_col("accuracy: {}%".format(successes / float(NUM_EPISODES) * 100.0), 'FG_YELLOW_BRIGHT')
    print_col("  - episodes:  {}".format(colorize(str(NUM_EPISODES), 'FG_WHITE')),        'FG_WHITE')
    print_col("  - successes: {}".format(colorize(str(successes),    'FG_GREEN_BRIGHT')), 'FG_WHITE')
    print_col("  - fails:     {}".format(colorize(str(fails),        'FG_RED_BRIGHT')),   'FG_WHITE')
    
    if successes > 0:
        print_col("Mean steps: {}\n".format(np.mean(results['steps'])), 'FG_YELLOW_BRIGHT')



if __name__ == "__main__":
    DEBUG_ENABLED = False
    NUM_EPISODES = 1
    LEN_EPISODE = 150
    WRITE_ENABLE = False
    # FILE_NAME = "trajectory_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    FILE_NAME = "trajectory"
           

    file_path = os.path.join(os.path.dirname(__file__), "../data/trajectories/" + FILE_NAME + ".txt")
    main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, file_path, DEBUG_ENABLED)