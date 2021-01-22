#!/usr/bin/env python3

from src.panda_server import gym_interface
import gym, panda_gym
import sys
from math import pi


def policy(current_joints):
    """ POLICY DEFINITION """
    code = input("Insert Code: ")
    if code == '1':
        return (-1.688, -0.369, 2.081, -2.628, -2.341, 0.454, 0.323)
    elif code == '2':
        return (0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi)
    elif code == '3':
        return (0, -pi/4, 0, -pi/2, 0, pi/3, 0)
    else:
        return (0, 0, 0, 0, 0, 0, 0)
    

def training():
    """ GYM TRAINING """
    env = gym.make('PandaReach-v0', render=True)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # random action
        obs, reward, done, info = env.step(action)

    env.close()


def testing():
    """ GYM TESTING WITH REAL PANDA """
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Create gym interface for connect to panda
    interface = gym_interface(HOST, PORT)

    while True:
        # Get current joints
        current_joints = interface.getCurrentJoints()
        print("Current Joints: {}".format(current_joints))

        # Check done task
        done = input("Done? [y/n]: ")
        if done == 'y':
            interface.sendDone()
            break
        
        # Process goal joints and execute them
        interface.sendGoalJoints(policy(current_joints))



if __name__ == "__main__":
    # training()
    testing()