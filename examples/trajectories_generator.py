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
    def __init__(self):
        # demo attributes
        self.phase = 0          # 1=pre-grasp, 2=grasp, 3=close, 4=place
        self.timeStep = 0
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3]) # [panda -> gym] trasformation
        self.obj_width = 0.04         # [m]

        # load ai
        self._loadAI()

        # create gym environment
        self.env = gym.make(self.args.env_name, render=True)


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
        self.timeStep = 0

        # reset environment
        observation = self.env.reset()

        # get observation
        self._gym_obs(observation)

        # get scenario's goals
        self.goal = observation["desired_goal"]             # goal place
        self.objOnStart_pose = self.obs[3:6]                # goal grasp
        self.preGrasp_pose = self.objOnStart_pose.copy()    # pre-goal grasp
        self.preGrasp_pose[2] += 0.031                      # [m] above the obj

    
    def _gym_obs(self, observation):
        # get obs
        self.obs = observation["observation"]
        self.obs[6:9]= 0
        self.obs[11:]= 0

        # get current tcp pose
        self.current_pose = observation["observation"][:3]             # grip_pos

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state


    def getTargetInfo(self):
        # get action
        self._policy()
        action = self.actions.copy()
        action[:3] *= 0.05  # Correct with panda-gym (limit maximum change in position)

        target_posit = [0, 0, 0]
        for i in range(len(target_posit)):
            target_posit[i] = self.panda_to_gym[i] + self.current_pose[i] + action[i]
        
        target_orien = [0, 0, 0, 1]

        target_gripper = self.obj_width if action[3] < 0 else 0.08
        grasp =                       1 if action[3] < 0 else 0

        return target_posit + target_orien + [target_gripper, grasp]


    def _policy(self):
        # PRE-GRASP
        if self.phase == 1:
            if np.linalg.norm(self.current_pose - self.preGrasp_pose) >= 0.031 and self.timeStep <= 20:
                self.env.render()
                self.actions = [0, 0, 0, 0]
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.preGrasp_pose, self.o_mean_approach, self.o_std_approach, self.g_mean_approach, self.g_std_approach, self.args)
                    pi = self.actor_network_approach(input_tensor)
                    self.actions = pi.detach().cpu().numpy().squeeze()
                self.actions[3] = 1
            else:
                self.phase = 2
                self.timeStep = 0
            
        # GRASP
        if self.phase == 2: 
            if np.linalg.norm(self.current_pose - self.objOnStart_pose) >= 0.015 and self.timeStep < self.env._max_episode_steps:
                self.env.render()
                self.actions = [0, 0, 0, 0]
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.objOnStart_pose, self.o_mean_manipulate, self.o_std_manipulate, self.g_mean_manipulate, self.g_std_manipulate, self.args)
                    pi = self.actor_network_manipulate(input_tensor)
                self.actions = pi.detach().cpu().numpy().squeeze()
                self.actions[3] = 1
            else:
                self.phase = 3
                self.timeStep = 0
   
        # PLACE
        if self.phase == 3:
            if np.linalg.norm(self.goal - self.current_pose) >= 0.031 and self.timeStep < self.env._max_episode_steps:
                self.env.render()
                self.actions = [0, 0, 0, 0]                    
                with torch.no_grad():
                    input_tensor = process_inputs(self.obs, self.goal, self.o_mean_retract, self.o_std_retract, self.g_mean_retract, self.g_std_retract, self.args)
                    pi = self.actor_network_retract(input_tensor)
                    self.actions = pi.detach().cpu().numpy().squeeze()
                self.actions[3] = -1
            else:
                self.phase = 0

        
    def step(self):
        observation, reward, done, self.info = self.env.step(self.actions)
        self._gym_obs(observation)
        self.timeStep += 1


    def checkGoal(self):
        return (self.phase == 0, self.info['is_success'])


    def __del__(self):
        # close gym environment
        self.env.close()



def main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, FILE_PATH):
    # for writing
    trajectory = list()

    # initialize Actor
    my_actor = PandaActor()

    # statistics
    results = {
        'successes': 0,
        'steps': list()
    }

    # start world
    for episode in range(NUM_EPISODES):
        # reset actory
        my_actor.reset()

        # reset trajectory
        trajectory.clear()
        
        # start episode
        for time_step in range(LEN_EPISODE):
            # generate a new action from observations and create a target pose with it
            target = my_actor.getTargetInfo()

            # add target to trajectory
            trajectory.append(target)

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
                    for e in info:
                        file_writer.write("{:>25}  ".format(e))
                    file_writer.write("\n")
    

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
    NUM_EPISODES = 2
    LEN_EPISODE = 150
    WRITE_ENABLE = True
    FILE_NAME = "trajectory_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
           

    file_path = os.path.join(os.path.dirname(__file__), "../data/trajectories/" + FILE_NAME + ".txt")
    main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, file_path)