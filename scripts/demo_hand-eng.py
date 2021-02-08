#!/usr/bin/env python3

import sys
sys.path.append("../scripts/")

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import gym, panda_gym
import numpy as np



class PandaActor():
    def __init__(self):
        # create gym environment
        self.env = gym.make("PandaPickAndPlace-v0", render = True)
        # phase 1 = approch
        # phase 2 = manipulate
        # phase 3 = retract
        self.phase = 0
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3])
        # self.panda_to_gym = np.array([-0.6918936446121056, -0.7441217819549181, -0.29851902093534083])
    
    def __del__(self):
        # close gym environment
        self.env.close()
    
    
    def reset(self):
        # start to do the demo
        observation = self.env.reset()
        self.goal = observation["desired_goal"]
        self.tcp_pose = observation["observation"][:3]
        self.object_pose = observation["observation"][3:6]
        self.object_rel_pose = observation["observation"][6:9]
        self.gripper_state = observation["observation"][9]
        # select first phase
        self.phase = 1
        return self.goal
        

    def _getAction(self):
        self.env.render()
        offset = 6
        if self.phase == 1: #approch
            self.object_oriented_goal = self.object_rel_pose.copy()
            self.object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object
            if np.linalg.norm(self.object_oriented_goal) >= 0.005:
                action = [0, 0, 0, 0]
                action[0] = self.object_oriented_goal[0] * offset
                action[1] = self.object_oriented_goal[1] * offset
                action[2] = self.object_oriented_goal[2] * offset
                action[3] = 1 # open gripper
                return action
            else:
                self.phase = 2
        
        if self.phase == 2: #manipulate
            if np.linalg.norm(self.object_rel_pose) >= 0.005:
                action = [0, 0, 0, 0]
                action[0] = self.object_rel_pose[0] * offset
                action[1] = self.object_rel_pose[1] * offset
                action[2] = self.object_rel_pose[2] * offset
                action[3] = -1 # close gripper
                return action
            else:
                self.phase = 3

        if self.phase == 3: #retract
            if np.linalg.norm(self.goal - self.object_pose) >= 0.01:
                action = [0, 0, 0, 0]
                action[0] = (self.goal[0] - self.object_pose[0]) * offset
                action[1] = (self.goal[1] - self.object_pose[1]) * offset
                action[2] = (self.goal[2] - self.object_pose[2]) * offset
                action[3] = -1 # close gripper
                return action
            else:
                self.phase = 0
        
        if self.phase == 0: # limit the number of timesteps in the episode to a fixed duration
            action = [0, 0, 0, -1] # close gripper
            return action


    def executeStep(self):
        # Get Observation
        print_col("CURRENT POSE:", 'FG_CYAN')
        print_col("[gym_to_tcp   ]: {}".format(self.tcp_pose), 'FG_DEFAULT')
        panda_current_pose = self.panda_to_gym + self.tcp_pose
        print_col("[panda_to_tcp ]: {}".format(panda_current_pose), "FG_BLUE")
        print_col("panda_to_tcp = panda_to_gym + gym_to_tcp", "FG_BLUE")

        # Get Action
        self.action = self._getAction()
        print_col("ACTION: {}\n".format(self.action), 'FG_CYAN')

        print_col("TARGET POSE:", 'FG_CYAN')
        print_col("[gym_to_target   ]: {}".format((self.tcp_pose + self.action[:3]) * 0.05), 'FG_DEFAULT')
        print_col("gym_to_target = (gym_to_tcp + action) * 0.05", 'FG_DEFAULT')
        panda_target_pose = (self.panda_to_gym + self.tcp_pose + self.action[:3]) * 0.05
        print_col("[panda_to_target ]: {}".format(panda_target_pose), "FG_BLUE")
        print_col("panda_to_target = (panda_to_gym + gym_to_tcp + action) * 0.05", "FG_BLUE")

        return self._step()
    

    def getNewPose(self, object_width, current_pose):
        # Get Observation
        print_col("CURRENT POSE:", 'FG_CYAN')
        print_col("[moveit_to_tcp]: {}".format(current_pose[:3]), 'FG_YELLOW_BRIGHT')
        print_col("[gym_to_tcp   ]: {}".format(self.tcp_pose), 'FG_DEFAULT')
        panda_current_pose = self.panda_to_gym + self.tcp_pose
        print_col("[panda_to_tcp ]: {}".format(panda_current_pose), "FG_BLUE")
        print_col("panda_to_tcp = panda_to_gym + gym_to_tcp", "FG_BLUE")
        if np.linalg.norm(panda_current_pose - np.array(current_pose[:3])) < 0.005:
            print("[Check        ]: " + colorize("True", 'FG_GREEN_BRIGHT') + "\n")
        else:
            print("[Check        ]: " + colorize("False", 'FG_RED_BRIGHT') + "\n")

        # Get Action
        self.action = self._getAction()
        print_col("ACTION: {}\n".format(self.action), 'FG_CYAN')

        # Perform action with real robot: moveit_to_tcp -> tcp_to_target 
        offset = 1.0 #6.0
        variation = 0.05
        x = current_pose[0] + ((self.action[0] / offset) * variation)
        y = current_pose[1] + ((self.action[1] / offset) * variation)
        z = current_pose[2] + ((self.action[2] / offset) * variation)
        if self.action[3] < 0:
            grip = object_width
            grasp = 1
        else:
            grip = 0.07 # gripper open
            grasp = 0
        new_pose = [x, y, z, 0, 0, 0, 1, grip, grasp]

        print_col("TARGET POSE:", 'FG_CYAN')
        print_col("[moveit_to_target]: {}".format(new_pose[:3]), 'FG_YELLOW_BRIGHT')
        print_col("moveit_to_target = moveit_to_tcp + action / 6.0 * 0.05", 'FG_YELLOW_BRIGHT')
        print_col("[gym_to_target   ]: {}".format((self.tcp_pose + self.action[:3]) * 0.05), 'FG_DEFAULT')
        print_col("gym_to_target = (gym_to_tcp + action) * 0.05", 'FG_DEFAULT')
        panda_target_pose = (self.panda_to_gym + self.tcp_pose + self.action[:3]) * 0.05
        print_col("[panda_to_target ]: {}".format(panda_target_pose), "FG_BLUE")
        print_col("panda_to_target = (panda_to_gym + gym_to_tcp + action) * 0.05", "FG_BLUE")
        if np.linalg.norm(panda_target_pose - np.array(new_pose[:3])) < 0.005:
            print("[Check        ]: " + colorize("True", 'FG_GREEN_BRIGHT') + "\n")
        else:
            print("[Check        ]: " + colorize("False", 'FG_RED_BRIGHT') + "\n")
        
        return new_pose, self._step()
    

    def _step(self):
        # put actions into the environment
        observation, reward, done, info = self.env.step(self.action)
        self.tcp_pose = observation["observation"][:3]
        self.gripper_state = observation["observation"][9]
        self.object_pose = observation["observation"][3:6]
        self.object_rel_pose = observation["observation"][6:9]
        return self.phase == 0



def testing_with_panda(panda):
    """GYM TESTING OFFLINE"""
    NUM_SCEN = 1
    LEN_SCEN = 150
    OBJECT_WIDTH = 0.02

    # Initialize Actor
    my_actor = PandaActor()

    # Initialize real robot:
    panda.getCurrentState()

    for i in range(NUM_SCEN):
        # Generate goal
        goal = my_actor.reset()
        print_col("[GYM] Scenario's goal: {}".format(goal), 'FG_MAGENTA')

        # Go sto start pose
        # j -0.001816946231006284 0.18395769805291534 0.0019670806476780292 -1.9086118257739941 -0.000429905939813149 2.095203423532922 0.7857393046819857 0.0
        start_pose = [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916, -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569, 0.07, 0] # Start pose gym
        # p 0.6014990053878944 1.5880450818915202e-06 0.29842061906465916 -3.8623752044513406e-06 -0.0013073068882995874 -5.91084615330739e-06 0.9999991454490569 0.00017145215566270055 0
        panda.sendGoalState(start_pose) 

        # Get current pose of Panda
        current_pose = panda.getCurrentState()
        print_col("[Moveit] Start Pose: {}\n".format(current_pose), 'FG_MAGENTA')

        # Catch moveit error
        if current_pose == "error":
            print_col("Abort", 'FG_RED')
            return

        for t in range(LEN_SCEN):
            # Process goal and execute them
            panda_target, done = my_actor.getNewPose(OBJECT_WIDTH, current_pose)
                
            # Send goal 
            panda.sendGoalState(panda_target)

            # Get current pose of Panda
            current_pose = panda.getCurrentState()

            # Catch moveit error
            if current_pose == "error":
                print_col("Abort", 'FG_RED')
                return

            if done:
                print("FINISH IN {} STEP".format(t))
                break
            print_col("---------------------", 'FG_GREEN')
            if t == 2:
                break
        print_col("Scenario {} finish".format(i), 'FG_GREEN')

    # Close communication
    panda.sendClose()
    print_col("close communication", 'FG_GREEN')



def testing_without_panda():
    """GYM TESTING OFFLINE"""
    NUM_SCEN = 1
    LEN_SCEN = 150

    # Initialize Actor
    my_actor = PandaActor()


    for i in range(NUM_SCEN):
        # Generate goal
        goal = my_actor.reset()
        print_col("[GYM] Scenario's goal: {}".format(goal), 'FG_MAGENTA')

        for t in range(LEN_SCEN):
            # Process goal and execute them
            done = my_actor.executeStep()

            if done:
                print("FINISH IN {} STEP".format(t))
                break
            print_col("---------------------", 'FG_GREEN')
        print_col("Scenario {} finish".format(i), 'FG_GREEN')

    # Close communication
    print_col("close communication", 'FG_GREEN')





if __name__ == "__main__":
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000
    ENABLE_REAL_PANDA = True

    if ENABLE_REAL_PANDA:
        # Create panda interface for connect to panda
        panda = PandaInterface(HOST, PORT)
        # Testing offline with panda
        testing_with_panda(panda)

    else:
        # Testing without panda
        testing_without_panda()