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
        return (-1.688, -0.369, 2.081, -2.628, -2.341, 0.454, 0.323, 0.0, 0.0)
    elif code == '2':
        return (0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi, 0.01, 0.01)
    elif code == '3':
        return (0, -pi/4, 0, -pi/2, 0, pi/3, 0, 0.0, 0.0)
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


def testing():
    """GYM TESTING WITH REAL PANDA"""
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Create gym interface for connect to panda
    interface = PandaInterface(HOST, PORT)

    while True:
        # Get current joints
        current_joints = interface.getCurrentJoints()

        # Check error
        if current_joints == 'error':
            print("Panda Error!")
            break
        else:
            print("Current Joints: {}".format(current_joints))

        # Check close goal
        if isFinish():
            interface.sendClose()
            break
        
        # Process goal joints and execute them
        goal_joints = policy(current_joints)

        # Perform them
        interface.sendGoalJoints(goal_joints)



if __name__ == "__main__":
    # training()
    testing()