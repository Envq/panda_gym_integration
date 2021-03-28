#!/usr/bin/env python3

# Panda-gym and AI
import gym
import panda_gym

# Other
import numpy as np
from datetime import datetime
import sys, os

# panda_controller
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../panda_controller/scripts/src")))
from utils import transform
from colors import print_col, colorize

# Custom
from panda_actors import AiActor, HandEngActor, E2EActor



class GymEnvironment():
    def __init__(self, DEBUG_ENABLED, OBJECT_WIDTH, ACTOR):
        # attributes
        self.debug_enabled = DEBUG_ENABLED
        self.obj_width = OBJECT_WIDTH
        # panda_gym internally applies this adjustment to actions (in _set_action()), 
        # so you need to apply it here as well 
        self.panda_gym_action_correction = 0.05  # (limit maximum change in position)

        # trasformations
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym]

        # create gym environment
        self.env = gym.make("PandaPickAndPlace-v0", render=True)

        # create actor
        self.actor = ACTOR
        
        if type(ACTOR) == E2EActor:
            self.last_gripper_state = 0.08  # open gripper
                    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def reset(self):
        # reset environment
        observation = self.env.reset()

        # get goal pose
        goal_posit = observation["desired_goal"]
        goal_orien = [0, 0, 0, 1]
        self.gym_to_goal = np.concatenate((goal_posit, goal_orien), axis=None)

        # get object pose on start
        objOnStart_posit = observation["observation"][3:6]         # object_pos
        objOnStart_orien = [0, 0, 0, 1]
        self.gym_to_objOnStart = np.concatenate((objOnStart_posit, objOnStart_orien), axis=None)
        
        # generate pre_grasp pose
        self.gym_to_preGrasp = self.gym_to_objOnStart.copy()
        self.gym_to_preGrasp[2] += 0.031  # [m] above the obj

        # reset attributes
        self.actor.reset(self.gym_to_goal, self.gym_to_objOnStart, self.gym_to_preGrasp)

        # get observation
        self._getObs(observation)

        # debug
        self._debugPrint("[panda] Obj pose: {}".format(transform(self.panda_to_gym, self.gym_to_objOnStart).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Goal pose: {}\n".format(transform(self.panda_to_gym, self.gym_to_goal).tolist()), 'FG_BLUE')

        # return start behaviour
        return [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916,  \
                -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569, \
                0.08] # open gripper

    
    def _getObs(self, observation):
        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.gym_to_current = np.concatenate((current_posit, current_orien), axis=None)

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state

        # get gym obs
        self.gym_obs = observation["observation"]


    def getTarget(self):
        self.env.render()

        # get action
        self.action = self._getAction()

        # process action
        action = self.action.copy()
        action[:3] *= self.panda_gym_action_correction              # panda gym correction
        action[7] = self.obj_width if action[7] < 0 else 0.08       # gripper width correction

        # generate target pose
        panda_to_current = transform(self.panda_to_gym, self.gym_to_current)
        panda_to_target = transform(panda_to_current, action[:7])

        self._debugPrint("[panda] Current pose: {}".format(panda_to_current.tolist()), 'FG_WHITE')
        self._debugPrint("[base ] Action: {}".format(self.action.tolist()), 'FG_WHITE')
        self._debugPrint("[final] Action: {}".format(action.tolist()), 'FG_WHITE')
        self._debugPrint("[panda] Target pose: {}\n".format(panda_to_target.tolist()), 'FG_WHITE')

        return panda_to_target.tolist() + [action[7]]
    

    def _getAction(self):
        # get fingers
        finger0 = finger1 = self.current_gripper / 2.0

        # generate obs for AI
        obs = np.zeros(25)
        obs[:3]    = self.gym_to_current[:3]     # grip_pos
        obs[3:6]   = self.gym_to_objOnStart[:3]  # object_pos
        obs[6:9]   = 0                         # object_rel_pos
        obs[9:11]  = [finger0, finger1]        # gripper_state
        obs[11:14] = 0                         # object_rot
        obs[14:17] = 0                         # object_velp
        obs[17:20] = 0                         # object_velr
        obs[20:23] = 0                         # grip_velp
        obs[23:25] = 0                         # gripper_vel

        # adjust object_pos info for retract-ai (It use object_pos as current_pose info)
        if type(self.actor) == AiActor and self.actor.getPhase() == 3:
            obs[3:6] = self.gym_to_current[:3]   # object_pos

        # adjust obs for E2EActor
        if type(self.actor) == E2EActor:
            obs = self.gym_obs

        # get action
        return self.actor.getAction(obs, self.gym_to_current, self.current_gripper)


    def step(self):
        # get correct action for gym
        pos = self.action[:3].tolist()
        grip = [self.action[7]]

        # insert action in gym and observe
        observation, reward, done, self.info = self.env.step(pos + grip)
        self._getObs(observation)


    def checkGoal(self):
        if self.actor.goalIsAchieved():
            # get statistics
            goal_pose  = transform(self.panda_to_gym, self.gym_to_goal)
            end_pose = transform(self.panda_to_gym, self.gym_to_current)
            
            stats = dict()
            stats['position_error'] = np.linalg.norm(goal_pose[:3] - end_pose[:3])
            stats['orientation_error'] = np.linalg.norm(goal_pose[3:] - end_pose[3:])
            stats['gym_success'] = self.info['is_success']
            return (True, stats)
            
        else:
            return (False, None)


    def __del__(self):
        # close gym environment
        self.env.close()



def main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, FILE_PATH, DEBUG_ENV_ENABLED, ACTOR, OBJECT_WIDTH):
    # for writing
    path = list()

    # initialize Actor
    my_actor = GymEnvironment(DEBUG_ENV_ENABLED, OBJECT_WIDTH, ACTOR)

    # statistics
    results = {
        'goals_achived': 0,
        'gym_successes': 0,
        'position_errors': list(),
        'orientation_errors': list(),
        'steps': list()
    }

    # start world
    for episode in range(NUM_EPISODES):
        # reset actory
        target = my_actor.reset()

        # reset path
        path.clear()

        # add start pose
        path.append(target)
        
        # start episode
        for time_step in range(LEN_EPISODE):
            # generate a new action from observations and create a target with it
            target  = my_actor.getTarget()

            # add target to path
            path.append(target)

            # perform a step and get new observations
            my_actor.step()

            # check goal
            (goal_achived, stats) = my_actor.checkGoal()

            # check the output condition
            if goal_achived:
                # add open gripper for E2E
                if type(ACTOR) == E2EActor:
                    open_gripper = path[-1].copy()
                    open_gripper[7] = 0.08
                    path.append(open_gripper)
                # get stats
                results['goals_achived'] += 1
                results['gym_successes'] += stats['gym_success']
                results['position_errors'].append(stats['position_error'])
                results['orientation_errors'].append(stats['orientation_error'])
                results['steps'].append(time_step + 1)
                break

        if goal_achived:
            print_col("[Episode {}] Goal achived in {} steps".format(episode, time_step + 1), 'FG_GREEN_BRIGHT')
        else:
            print_col("[Episode {}] Goal not achived in {} steps".format(episode, time_step + 1), 'FG_RED_BRIGHT')
        
        # write to file
        if WRITE_ENABLE and input("Write to file? [y/n] ") == 'y':
            with open(FILE_PATH, 'w') as file_writer:
                for point in path:
                    for e in point:
                        file_writer.write("{:>25}  ".format(e))
                    file_writer.write("\n")
    

    # Generate final statistics
    print("-----------------------------------")
    print_col("All Episodes finish", 'FG_YELLOW_BRIGHT')

    successes = results['goals_achived']
    # successes = results['gym_successes']
    fails = NUM_EPISODES - successes
    percentage = successes / float(NUM_EPISODES) * 100.0
    if percentage == 100:
        percentage = colorize(str(percentage) + '%', 'FG_GREEN')
    else:
        percentage = colorize(str(percentage) + '%', 'FG_RED')
    print_col("accuracy:      {}".format(percentage), 'FG_YELLOW_BRIGHT')
    print_col("  - episodes:  {}".format(NUM_EPISODES), 'FG_WHITE')
    print_col("  - successes: {}".format(successes), 'FG_WHITE')
    print_col("  - fails:     {}".format(fails), 'FG_WHITE')
    print_col("  - gym successes: {}".format(results['gym_successes']), 'FG_WHITE')
    
    if successes > 0:
        print_col("Mean position errors:    {}"  .format(np.mean(results['position_errors'])), 'FG_YELLOW_BRIGHT')
        print_col("Mean orientation errors: {}"  .format(np.mean(results['orientation_errors'])), 'FG_YELLOW_BRIGHT')
        print_col("Mean steps:              {}\n".format(np.mean(results['steps'])), 'FG_YELLOW_BRIGHT')



if __name__ == "__main__":
    """Configure script here..."""
    DEBUG_ENV_ENABLED = False
    DEBUG_AI_ENABLED = False
    NUM_EPISODES = 1
    LEN_EPISODE = 150
    WRITE_ENABLE = True
    # FILE_NAME = "path_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    FILE_NAME = "path"

    ACTOR = AiActor(DEBUG_ENABLED=DEBUG_AI_ENABLED, MAX_EPISODE_STEPS = 50)
    # ACTOR = E2EActor(DEBUG_ENABLED=DEBUG_AI_ENABLED, MAX_EPISODE_STEPS = 50)
    # ACTOR = HandEngActor(DEBUG_ENABLED=DEBUG_AI_ENABLED, MAX_EPISODE_STEPS = 50)

    OBJECT_WIDTH = 0.04  # [m]
           

    file_path = os.path.join(os.path.dirname(__file__), "../data/paths/" + FILE_NAME + ".txt")
    main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, file_path, DEBUG_ENV_ENABLED, ACTOR, OBJECT_WIDTH)