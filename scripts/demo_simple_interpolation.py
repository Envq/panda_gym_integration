#!/usr/bin/env python3

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import gym, panda_gym
import numpy as np
import time
import sys


class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, debug_mode, render, enable_real_panda, HOST, PORT):
        # demo parameters
        self.enable_real_panda = enable_real_panda
        self.debug_mode = debug_mode
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3]) # [panda -> gym] trasformation
        # self.panda_to_gym = np.array([-0.6918936446121056, -0.7441217819549181, -0.29851902093534083])
        self.tolerance = 0.005        # [m]
        self.phase_change_delay = 1   # [sec]
        self.obj_width = 0.04         # [m]
        self.gripper_move_steps = 10  # [step]
        self.phase = 0  # 1=pre-grasp, 2=grasp, 3=close, 4=place
        self.offset = 6 # decrease this value to increase the points of the trajectory

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
    

    def goalAchieved(self):
        return self.phase == 0

    
    def reset(self):
        self.phase = 1                                                      # select first phase
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
        self.action = self._actor_getAction(self.gym_to_tcp, self.actor_fingersWidth)                           # get action

        self._debugPrint("action: {}".format(self.action.tolist()), 'FG_MAGENTA')
        # print_col("action final: {}".format((self.action * 0.05).tolist()), 'FG_MAGENTA')
        # self._debugPrint("[gym  ] Target: {}".format((self.gym_to_tcp + self.action[:3]).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Target: {}".format((self.panda_to_gym + self.gym_to_tcp + self.action[:3]).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Target final: {}".format((self.panda_to_gym + self.gym_to_tcp + self.action[:3]*0.05).tolist()), 'FG_BLUE')

        if self.enable_real_panda:
            # Note: here I use panda_to_tcp to reduce the error between panda_gym and moveit caused by not reaching the target_pose after one step
            self.real_to_target = self._panda_get_target(self.panda_to_tcp, self.actor_fingersWidth, self.action)  # get target pose
            
            self._debugPrint("[real ] Target: {}".format(self.real_to_target), 'FG_BLUE')
        self._debugPrint("", 'FG_DEFAULT')
    
    
    def step(self):
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

        # get immutable informations
        self.gym_to_objOnStart = observation["observation"][3:6]   # object_pos
        self.goal = observation["desired_goal"]

        self.pre_grasp_goal = self.gym_to_objOnStart.copy()
        self.pre_grasp_goal[2] += 0.10  # [m] above the obj

        # Get and return observations (tcp + gripper width)
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        fingersWidth = finger0 + finger1
        gym_to_tcp = observation["observation"][:3]             # grip_pos
        return gym_to_tcp, fingersWidth
        

    def _actor_getAction(self, gym_to_tcp, actor_fingersWidth):
        # SETTINGS
        self.env.render()
        offset = self.offset 

        # PRE-GRASP APPROCH
        if self.phase == 1:
            if np.linalg.norm(self.pre_grasp_goal - gym_to_tcp) >= self.tolerance:
                action = [0, 0, 0, 0]
                action[0] = (self.pre_grasp_goal[0] - gym_to_tcp[0]) * offset
                action[1] = (self.pre_grasp_goal[1] - gym_to_tcp[1]) * offset
                action[2] = (self.pre_grasp_goal[2] - gym_to_tcp[2]) * offset
                action[3] = 1 # open gripper
                return np.array(action)
            else:
                self.phase = 2
                print_col("PRE-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # GRASP APPROCH
        if self.phase == 2: 
            if np.linalg.norm(self.gym_to_objOnStart - gym_to_tcp) >= self.tolerance: 
                action = [0, 0, 0, 0]
                action[0] = (self.gym_to_objOnStart[0] - gym_to_tcp[0]) * offset
                action[1] = (self.gym_to_objOnStart[1] - gym_to_tcp[1]) * offset
                action[2] = (self.gym_to_objOnStart[2] - gym_to_tcp[2]) * offset
                action[3] = 1 # open gripper
                return np.array(action)
            else:
                self.phase = 3
                self.timer = 0
                print_col("GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # CLOSE GRIPPER
        if self.phase == 3: 
            if self.timer < self.gripper_move_steps:
                action = [0, 0, 0, -1] # close gripper
                self.timer += 1
                return np.array(action)
            else:
                self.phase = 4
                self.timer = 0
                print_col("GRIPPER CLOSE: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)

        # PLACE APPROCH
        if self.phase == 4:
            if np.linalg.norm(self.goal - gym_to_tcp) >= self.tolerance:
                action = [0, 0, 0, 0]
                action[0] = (self.goal[0] - gym_to_tcp[0]) * offset
                action[1] = (self.goal[1] - gym_to_tcp[1]) * offset
                action[2] = (self.goal[2] - gym_to_tcp[2]) * offset
                action[3] = -1 # close gripper
                return np.array(action)
            else:
                self.phase = 0
                print_col("POST-GRASP: successful", 'FG_YELLOW_BRIGHT')
                time.sleep(self.phase_change_delay)
        
        if self.phase == 0: # limit the number of timesteps in the episode to a fixed duration
            action = [0, 0, 0, -1] # close gripper
            return np.array(action)


    def _actor_step(self, action):
        # put actions into the environment and get observations
        observation, reward, done, info = self.env.step(action)
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
        offset = self.offset 
        variation = 0.05 # This is the correction of pos_ctrl in line 81 panda_env.py in panda_gym

        x = real_to_tcp[0] + ((action[0] / offset) * variation)
        y = real_to_tcp[1] + ((action[1] / offset) * variation)
        z = real_to_tcp[2] + ((action[2] / offset) * variation)
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