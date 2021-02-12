#!/usr/bin/env python3

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import gym, panda_gym
import numpy as np
import time
import sys


"""
TO DO PandaActor with:
- __init__()            -> manage your variables
- reset()               -> ad
- getAction()           -> manage your method of generating action
- step()                -> manage your method of generation observations
- goalAchieved()        -> manage your method of controlling the achievement of the goal

- _actor_reset()        -> add custom observations and intermediate goals
- _actor_getAction()    -> create your policy
- _actor_step()         -> add custom observations


"""


class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, debug_mode, render, enable_real_panda, HOST, PORT):
        """

        edit it with your custom attributes

        """
        # demo parameters
        self.enable_real_panda = enable_real_panda
        self.debug_mode = debug_mode
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3]) # [panda -> gym] trasformation
        # self.panda_to_gym = np.array([-0.6918936446121056, -0.7441217819549181, -0.29851902093534083])
        self.tolerance = 0.005        # [m]

        # initialize
        self._actor_init(scenario="PandaPickAndPlace-v0", render=render)

        if self.enable_real_panda:
            print_col("==================================================", 'FG_GREEN')
            self._panda_init(HOST, PORT)
            print_col("==================================================", 'FG_GREEN')
        
        self._debugPrint("panda -> gym: {}".format(self.panda_to_gym.tolist()), 'FG_BLUE')
    
    
    def _debugPrint(self, msg, color):
        if self.debug_mode: 
            print_col(msg, color)
    
    
    def reset(self):
        """

        Edit this with reset actions, and debug comments

        """
        self.gym_to_tcp, self.actor_fingersWidth = self._actor_reset()      # reset actor and get start pose
        self.panda_to_tcp = self.panda_to_gym + self.gym_to_tcp             # update panda_to_tcp
        
        self._debugPrint("[gym  ] Goal: {}".format(self.goal.tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Goal: {}\n".format((self.panda_to_gym + self.goal).tolist()), 'FG_BLUE')

        self._debugPrint("[gym  ] Start: {}".format(self.gym_to_tcp.tolist()   + [self.actor_fingersWidth]), 'FG_BLUE')
        self._debugPrint("[panda] Start: {}".format(self.panda_to_tcp.tolist() + [self.actor_fingersWidth]), 'FG_BLUE')

        if self.enable_real_panda:
            self.real_to_tcp, self.real_fingersWidth = self._panda_reset()            # reset real-panda and get start pose
            
            self._debugPrint("[real ] Start: {}\n".format(self.real_to_tcp.tolist() + [self.real_fingersWidth]), 'FG_WHITE')
            if np.linalg.norm(self.panda_to_tcp - self.real_to_tcp) < self.tolerance:
                print("Check Start Pose: " + colorize("True", 'FG_GREEN_BRIGHT') + "\n")
            else:
                print("Check Start Pose: " + colorize("False", 'FG_RED_BRIGHT') + "\n")
        self._debugPrint("", 'FG_DEFAULT')
        

    def getAction(self):
        """

        Edit this with your methods for generate target_pose for real robot

        """
        self.action = self._actor_getAction(self.gym_to_tcp, self.actor_fingersWidth)                           # get action

        self._debugPrint("action: {}".format(self.action.tolist()), 'FG_MAGENTA')
        # print_col("action final: {}".format((self.action * 0.05).tolist()), 'FG_MAGENTA')
        # self._debugPrint("[gym  ] Target: {}".format((self.gym_to_tcp + self.action[:3]).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Target: {}".format((self.panda_to_gym + self.gym_to_tcp + self.action[:3]).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Target final: {}".format((self.panda_to_gym + self.gym_to_tcp + self.action[:3]*0.05).tolist()), 'FG_BLUE')

        if self.enable_real_panda:
            self.real_to_target = self._panda_get_target(self.real_to_tcp, self.real_fingersWidth, self.action)  # get target pose
            
            self._debugPrint("[real ] Target: {}".format(self.real_to_target), 'FG_BLUE')
        self._debugPrint("", 'FG_DEFAULT')
    
    
    def step(self):
        """

        Edit this with your methods for generate target_pose for real robot

        """
        self.gym_to_tcp, self.actor_fingersWidth = self._actor_step(self.action)            # perform a step and get the new current pose
        self.panda_to_tcp = self.panda_to_gym + self.gym_to_tcp                             # update panda_to_tcp
        
        # self._debugPrint("[gym  ] Current: {}".format(self.gym_to_tcp.tolist()   + [self.actor_fingersWidth]), 'FG_BLUE')
        self._debugPrint("[panda] Current: {}".format(self.panda_to_tcp.tolist() + [self.actor_fingersWidth]), 'FG_BLUE')

        if self.enable_real_panda:
            self.real_to_tcp,  self.real_fingersWidth = self._panda_step(self.real_to_target) # move real-panda and get the new current pose
            
            self._debugPrint("[real ] Current: {}\n".format( self.real_to_tcp.tolist() + [self.real_fingersWidth]), 'FG_WHITE')
            if np.linalg.norm(self.panda_to_tcp -  self.real_to_tcp) < self.tolerance:
                print("Check Current Pose: " + colorize("True", 'FG_GREEN_BRIGHT') + "\n")
            else:
                print("Check Current Pose: " + colorize("False", 'FG_RED_BRIGHT') + "\n")
        self._debugPrint("", 'FG_DEFAULT')


    def goalAchieved(self):
        """

        Edit this with you goal achived control

        """
        goalAchieved = False
        return goalAchieved


    def __del__(self):
        self._actor_del()

        if self.enable_real_panda:
            print_col("close communication", 'FG_MAGENTA')
            self._panda_del()

    
    """ACTOR BEHAVIOUR"""
    def _actor_init(self, scenario, render):
        # create gym environment
        self.env = gym.make(scenario, render)

        
    def _actor_reset(self):
        # start to do the demo
        observation = self.env.reset()
        """

        edit with for add custom observations and intermediate goals (pre_grasp_pose, object_pose ecc)

        """
        # Get and return observations (tcp + gripper width)
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        fingersWidth = finger0 + finger1
        gym_to_tcp = observation["observation"][:3]             # grip_pos
        return gym_to_tcp, fingersWidth
        

    def _actor_getAction(self, gym_to_tcp, actor_fingersWidth):
        # SETTINGS
        self.env.render()
        """
        
        Edit this with your policy

        """
        action = [0, 0, 0, -1] # close gripper
        return np.array(action)


    def _actor_step(self, action):
        # put actions into the environment and get observations
        observation, reward, done, info = self.env.step(action)
        """

        edit with for add custom observations

        """
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        fingersWidth = finger0 + finger1
        gym_to_tcp = observation["observation"][:3]
        return gym_to_tcp, fingersWidth
    

    def _actor_del(self):
        # close gym environment
        self.env.close()
    
    
    """REAL PANDA BEHAVIOUR"""
    def _panda_init(self, HOST, PORT):
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection
        

    def _panda_reset(self):
        # Go to the start pose
        # j -0.001816946231006284 0.18395769805291534 0.0019670806476780292 -1.9086118257739941 -0.000429905939813149 2.095203423532922 0.7857393046819857 0.0
        real_to_start = [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916, -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569, 0.08, 0] # Start pose gym
        self.panda.sendGoalState(real_to_start) 

        # Get current pose of Panda
        current_pose = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_pose == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()

        # Get and return observations (tcp + gripper width)
        return np.array(current_pose[:3]), current_pose[3]
        

    def _panda_get_target(self, real_to_tcp, real_fingersWidth, action):
        # Perform action with real robot: real_to_tcp -> tcp_to_target
        variation = 0.05 # This is the correction of pos_ctrl in line 81 panda_env.py in panda_gym

        x = real_to_tcp[0] + (action[0] * variation)
        y = real_to_tcp[1] + (action[1] * variation)
        z = real_to_tcp[2] + (action[2] * variation)
        if action[3] < 0:
            grip = self.obj_width    # gripper close to obj
            grasp = 1
        else:
            grip = 0.07              # gripper open
            grasp = 0
        real_to_target = [x, y, z, 0, 0, 0, 1, grip, grasp]

        return real_to_target
    

    def _panda_step(self, real_to_target):
        # Send goal 
        self.panda.sendGoalState(real_to_target)

        # Get current pose of Panda
        current_pose = self.panda.getCurrentState()

        # Catch moveit error
        if current_pose == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()

        # Get and return observations (tcp + gripper width)
        return np.array(current_pose[:3]), current_pose[3]
    

    def _panda_del(self):
        # Close communication
        self.panda.sendClose()



def main(NUM_EPISODES, LEN_EPISODE, DEBUG_MODE, RENDER, ENABLE_REAL_PANDA, HOST, PORT):
    # initialize Actor
    my_actor = PandaActor(DEBUG_MODE, RENDER, ENABLE_REAL_PANDA, HOST, PORT)

    for episode in range(NUM_EPISODES):
        # reset actor and get first observations
        my_actor.reset()

        goal_achived = False
        for time_step in range(LEN_EPISODE):
            if DEBUG_MODE:
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
    DEBUG_MODE = True
    RENDER = True
    ENABLE_REAL_PANDA = False
    NUM_EPISODES = 1
    LEN_EPISODE = 100
    LIMIT_STEP = LEN_EPISODE

    if (len(sys.argv) > 1):
        if sys.argv[1] == 'real':
            ENABLE_REAL_PANDA = True

        if len(sys.argv) > 2:
            LEN_EPISODE = int(sys.argv[2])


    main(NUM_EPISODES, LEN_EPISODE, DEBUG_MODE, RENDER, ENABLE_REAL_PANDA, HOST, PORT)