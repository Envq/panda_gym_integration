#!/usr/bin/env python3

from src.panda_server import PandaInterface
import gym, panda_gym
from math import pi



def isFinish():
    """RETURN IF TASK IS FINISHED"""
    done = input("Done? [y/n]: ")
    return done == 'y'


def policy(current_joints):
    """POLICY DEFINITION"""
    code = input("Insert Code: ")
    if code == '1':
        return (0.307019570052, -5.22132961561e-12, 0.590269558277, 0.923955699469, -0.382499497279, 1.3249325839e-12, 3.20041176635e-12, 0.07)
    elif code == '2':
        return (0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.00)
    elif code == '3':
        return (0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.01)
    elif code == '4':
        return (0.4, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.70)
    else:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    

def training():
    """GYM TRAINING"""
    env = gym.make('PandaReach-v0', render=True)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # random action
        obs, reward, done, info = env.step(action)

    env.close()


def testing(HOST, PORT):
    """GYM TESTING WITH REAL PANDA"""

    # Create panda interface for connect to panda
    panda = PandaInterface(HOST, PORT)

    while True:
        # Get current pose
        current_pose = panda.getCurrentState()
        print("Current Pose: {}".format(current_pose))

        # Check error
        if current_pose == 'error':
            break

        # Check close goal
        if isFinish():
            panda.sendClose()
            break
        
        # Process goal joints and execute them
        goal_pose = policy(current_pose)

        # Perform them
        panda.sendGoalState(goal_pose)



if __name__ == "__main__":
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # training()
    
    testing(HOST, PORT)