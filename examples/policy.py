#!/usr/bin/env python3

import sys
sys.path.append("../scripts/")

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import numpy as np



class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, debug_mode, HOST, PORT):
        # demo parameters
        self.debug_mode = debug_mode
        self.tolerance = 0.005     # [m]
        self.timer = 0

        # initialize
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection
    
    
    def _debugPrint(self, msg, color):
        if self.debug_mode: 
            print_col(msg, color)


    def goalAchieved(self):
        return np.linalg.norm(self.goal_pose - self.current_pose) < self.tolerance
    
    
    def reset(self):
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

        self._debugPrint("[real] Start: {}".format(self.current_pose.tolist() + [self.current_gripper]), 'FG_BLUE')
        

    def getAction(self):
        self.target_pose = self.current_pose.copy()
        if self.timer < 5:
            self.target_pose[0] += 0.01
        else:
            self.target_pose[0] -= 0.01
        self.target_gripper = self.current_gripper

        self._debugPrint("[real] Target: {}".format(self.target_pose.tolist() + [self.target_gripper]), 'FG_BLUE')
    
    
    def step(self):
        self.timer += 1

        # Send goal
        self.panda.sendGoalState(self.target_pose.tolist() + [self.target_gripper, 0]) # gripper_open

        # Get current pose and gripper state of Panda
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()
            
        self.current_pose = np.array(current_msg[:7])
        self.current_gripper = current_msg[7] # fingers width

        self._debugPrint("[real] Current: {}".format(self.current_pose.tolist() + [self.current_gripper]), 'FG_BLUE')


    def __del__(self):
        print_col("close communication", 'FG_MAGENTA')
        self.panda.sendClose()



def main(NUM_EPISODES, LEN_EPISODE, DEBUG_MODE, HOST, PORT):
    # initialize Actor
    my_actor = PandaActor(DEBUG_MODE, HOST, PORT)

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
    NUM_EPISODES = 1
    LEN_EPISODE = 100
    DEBUG_MODE = False

    if (len(sys.argv) > 1):
        if sys.argv[1] == 'debug':
            DEBUG_MODE = True

        if len(sys.argv) > 2:
            LEN_EPISODE = int(sys.argv[2])


    main(NUM_EPISODES, LEN_EPISODE, DEBUG_MODE, HOST, PORT)