#!/usr/bin/env python3

from src.panda_server import PandaInterface
from src.colors import print_col, colorize
import gym, panda_gym
import numpy as np
import sys



class PandaActor():
    """GLOBAL BEHAVIOUR"""
    def __init__(self, debug_mode, HOST, PORT):
        """
        Edit this with your custom attributes and initializations
        """
        # demo parameters
        self.debug_mode = debug_mode
        #current_msg -> Position(x, y, z)  Orientation(x, y, z , w)  Fingers(width)
        #target_msg  -> Position(x, y, z)  Orientation(x, y, z , w)  Fingers(width)  Grasp
        #goal_pose   -> Position(x, y, z)  Orientation(x, y, z , w)

        # initialize
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection
    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_mode: 
            print_col(msg, color)


    def goalAchieved(self):
        """
        Edit this with your goal achived control
        """
        # return np.linalg.norm(self.goal_pose - self.current_pose) < 0.05
    
    
    def reset(self):
        """
        Edit this with goal and start msg and do first observations
        """
        self.goal_pose = np.zeros(7) #Initialize this...
        print_col("[real] Goal: {}".format(self.goal_pose.tolist()), 'FG_MAGENTA')

        start_msg = list() #Initialize this...

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
        """
        Edit this with your methods for generate target for real robot
        """
        #self.target_pose = self.current_pose + ...
        #self.target_gripper = ...
        #self.grasp = ...
    
    
    def step(self):
        """
        Edit this with your methods for generate target for real robot
        """
        # Send goal
        msg = self.panda.sendGoalState(self.target_pose.tolist() + [self.target_gripper, self.grasp])

        # Get current pose and gripper state of Panda
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()
            
        self.current_pose = np.array(current_msg[:7])
        self.current_gripper = current_msg[7] # fingers width

        self._debugPrint("[real] Start: {}".format(self.current_pose.tolist() + [self.current_gripper]), 'FG_BLUE')


    def __del__(self):
        """
        Edit this with your destructors
        """
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
    DEBUG_MODE = True
    NUM_EPISODES = 1
    LEN_EPISODE = 100
    LIMIT_STEP = LEN_EPISODE

    if (len(sys.argv) > 1):
        if sys.argv[1] == 'real':
            ENABLE_REAL_PANDA = True

        if len(sys.argv) > 2:
            LEN_EPISODE = int(sys.argv[2])


    main(NUM_EPISODES, LEN_EPISODE, DEBUG_MODE, HOST, PORT)