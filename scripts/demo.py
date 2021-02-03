#!/usr/bin/env python3

from src.panda_server import PandaInterface
from src.colors import print_col
import gym, panda_gym


class Actor():
    def __init__(self):
        self.sign = -1

    def _getObs(self, current_pose):
        # Get Observation: [p_x, p_y, p_z,  p_g1, p_g2,  v_x, v_y, v_z,  v_g1, v_g2]
        obs = [current_pose[0], current_pose[1], current_pose[2]]
        obs = obs + [current_pose[7]/2.0, current_pose[7]/2.0]
        obs = obs + [0, 0, 0, 0, 0]
        return obs   
    
    def _getAction(self, obs, time_step):
        if (time_step % 5) == 0:
            self.sign = - self.sign
        delta = self.sign * 0.01
        return [delta, 0, 0, 0]


    def getNewPose(self, current_pose, time_step):
        obs = self._getObs(current_pose)
        # print("obs: ", obs)

        action = self._getAction(obs, time_step)
        print("action: ", action)
        
        # Perform action with real robot:
        # world_to_gripper -> gripper_to_target
        x = obs[0] + action[0]
        y = obs[1] + action[1]
        z = obs[2] + action[2]
        grip = obs[3]*2 + action[3]*2
        grasp = 0
        new_pose = [x, y, z, 0, 0, 0, 1, grip, grasp]
        # print("panda new target pose: ", new_pose)
        return new_pose
    


def testing(panda):    
    """GYM TESTING"""
    NUM_SCEN = 1
    LEN_SCEN = 10

    # Initialize real robot:
    panda.getCurrentState()
    start_pose = [0.30686807928115, 4.6674387784271756e-05, 0.4867293030857287, 3.254063654416475e-05, 8.471364455930387e-05, 6.05906646632388e-05, 0.9999999940467379, 0.00017906040996313097, 0] # Ready pose
    panda.sendGoalState(start_pose) 

    # Get current pose of Panda
    current_pose = panda.getCurrentState()
    print_col("Start Pose: {}\n".format(current_pose), 'FG_MAGENTA')

    # Catch moveit error
    if current_pose == "error":
        print("Abort")
        return

    # Initialize Actor
    my_actor = Actor()

    for i in range(NUM_SCEN):
        # Generate goal
        g = [start_pose[0] + 0.01 * 4, start_pose[1], start_pose[2]]
        print_col("Scenario's goal: {}\n".format(g), 'FG_MAGENTA')

        for t in range(LEN_SCEN):
            # Process goal and execute them
            panda_target = my_actor.getNewPose(current_pose, t)
                
            # Send goal 
            panda.sendGoalState(panda_target)

            # Get current pose of Panda
            current_pose = panda.getCurrentState()
            print("Current Pose: ", current_pose)

            # Catch moveit error
            if current_pose == "error":
                print_col("Abort", 'FG_RED')
                return
            print("---------------------")
        print_col("Scenario {} finish".format(i), 'FG_GREEN')

    # Close communication
    panda.sendClose()
    print_col("close communication", 'FG_GREEN')



if __name__ == "__main__":
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Create panda interface for connect to panda
    panda = PandaInterface(HOST, PORT)

    # testing_offline(panda)
    testing(panda)