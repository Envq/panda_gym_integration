import torch
from rl_modules.models import actor
from rl_modules.arguments import get_args
import gym
import panda_gym
import numpy as np

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path_approach = args.save_dir + args.env_name + '/approach.pt'
    o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, model_approach = torch.load(model_path_approach, map_location=lambda storage, loc: storage)
    model_path_manipulate = args.save_dir + args.env_name + '/manipulate.pt'
    o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
    model_path_retract = args.save_dir + args.env_name + '/retract.pt'
    o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, model_retract = torch.load(model_path_retract, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name, render = True)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network_approach = actor(env_params)
    actor_network_approach.load_state_dict(model_approach)
    actor_network_approach.eval()
    actor_network_manipulate = actor(env_params)
    actor_network_manipulate.load_state_dict(model_manipulate)
    actor_network_manipulate.eval()
    actor_network_retract = actor(env_params)
    actor_network_retract.load_state_dict(model_retract)
    actor_network_retract.eval()
    success = 0

    for i in range(args.demo_length):
        observation = env.reset()
        obs = observation["observation"]

        goal = observation["desired_goal"]
        objectPos = obs[3:6]
        timeStep = 0
        grip_pos = obs[:3]
        object_pos_goal = objectPos.copy()
        object_pos_goal[2] += 0.031
        obs[6:9]= 0
        obs[11:]= 0

        
        while np.linalg.norm(grip_pos - object_pos_goal) >= 0.031 and timeStep <= 20:
            env.render()
            with torch.no_grad():
                #input_tensor = _preproc_inputs(obs, objectPos)
                input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args)
                pi = actor_network_approach(input_tensor)
                # convert the actions
                actions = pi.detach().cpu().numpy().squeeze()

            actions[3] = 0.05

            observation_new, reward, _, info = env.step(actions)
            print("action1: ", actions * 0.05)
            obs = observation_new["observation"]
            grip_pos = obs[:3]
            obs[6:9]= 0
            obs[11:]= 0

            timeStep += 1
        
        timeStep = 0
        'manipulate action'
        while np.linalg.norm(grip_pos - objectPos) >= 0.015 and timeStep < env._max_episode_steps:
            env.render()
            actions = [0, 0, 0, 0]
            with torch.no_grad():
                #input_tensor = _preproc_inputs(obs, objectPos)
                input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args)
                pi = actor_network_manipulate(input_tensor)
                # convert the actions
            actions = pi.detach().cpu().numpy().squeeze()
            
            actions[3] = 0.05
            
            observation_new, _, _, info = env.step(actions)
            print("action2: ", actions * 0.05)
            obs = observation_new["observation"]
            grip_pos = obs[:3]
            
            timeStep += 1
            
        timeStep = 0
        'retract actions'
        while np.linalg.norm(goal - grip_pos) >= 0.031 and timeStep < env._max_episode_steps:
            env.render()
                
            with torch.no_grad():
                input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args)
                pi = actor_network_retract(input_tensor)
                # convert the actions
                actions = pi.detach().cpu().numpy().squeeze()
                # convert the actions
         
            actions[3] = -1

            # put actions into the environment
            observation_new, reward, _, info = env.step(actions)
            print("action3: ", actions * 0.05)
            obs = observation_new['observation']
            grip_pos = obs[:3]
            obs[6:9]= 0
            obs[11:]= 0
    
            timeStep += 1
        
    

        while True: 
            env.render() 
            action = [0, 0, 0, 0]
            action[3] = -0.01  # keep the gripper closed

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew["observation"][3:6]
            object_rel_pos = obsDataNew["observation"][6:9]

            # limit the number of timesteps in the episode to a fixed duration
            if timeStep >= env._max_episode_steps:
                break

            
        #print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        if info['is_success']:
            success += 1

    print('Total success:', success, '%')