#!/usr/bin/env python3

# Custom
import sys
sys.path.append("../scripts/")
from src.utils import quaternion_multiply, transform
from src.colors import print_col, colorize

# Panda-gym and AI
import gym
import panda_gym
from ai.panda_actors import AiActor, HandEngActor, E2EActor

# Other
import numpy as np
from datetime import datetime
import os



class GymEnvironment():
    def __init__(self, DEBUG_ENABLED, ACTOR):
        # attributes
        self.debug_enabled = DEBUG_ENABLED
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym] trasformation
        self.last_phase = 0
        self.obj_width = 0.04                    # [m]
        # panda_gym internally applies this adjustment to actions (in _set_action()), 
        # so you need to apply it here as well 
        self.panda_gym_action_correction = 0.05  # (limit maximum change in position)

        # create gym environment
        self.env = gym.make("PandaPickAndPlace-v0", render=True)

        # create actor
        self.actor = ACTOR
        if type(ACTOR) == AiActor or type(ACTOR) == HandEngActor:
            self.actor.setMaxEpisodeSteps(self.env._max_episode_steps)
                    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def reset(self):
        # reset attributes
        self.last_phase = 1

        # reset environment
        observation = self.env.reset()

        # get goal pose
        goal_posit = observation["desired_goal"]
        goal_orien = [0, 0, 0, 1]
        self.goal_pose = np.concatenate((goal_posit, goal_orien), axis=None)

        # get object pose on start
        objOnStart_posit = observation["observation"][3:6]         # object_pos
        objOnStart_orien = [0, 0, 0, 1]
        self.objOnStart_pose = np.concatenate((objOnStart_posit, objOnStart_orien), axis=None)
        
        # generate pre_grasp pose
        self.preGrasp_pose = self.objOnStart_pose.copy()
        self.preGrasp_pose[2] += 0.031  # [m] above the obj

        # reset attributes
        self.actor.reset(self.goal_pose, self.objOnStart_pose, self.preGrasp_pose)

        # get observation
        self._getObs(observation)

        # debug
        self._debugPrint("[panda] Obj pose: {}".format(transform(self.panda_to_gym, self.objOnStart_pose).tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Goal pose: {}\n".format(transform(self.panda_to_gym, self.goal_pose).tolist()), 'FG_BLUE')

        # return start behaviour
        return ([0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916,  \
                -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569], \
                0.08) # open gripper

    
    def _getObs(self, observation):
        # get current tcp pose
        current_posit = observation["observation"][:3]             # grip_pos
        current_orien = [0, 0, 0, 1]
        self.current_pose = np.concatenate((current_posit, current_orien), axis=None)

        # get current gripper
        finger0 = observation["observation"][9]
        finger1 = observation["observation"][10]
        self.current_gripper = finger0 + finger1                   # gripper_state

        # get gym obs
        self.gym_obs = observation["observation"]


    def getTargetInfo(self):
        self.env.render()

        # get action
        self.action = self._getAction()

        # process action
        action = self.action.copy()
        action[:3] *= self.panda_gym_action_correction

        # generate gripper state
        if self.actor.getPhase() == 2 and self.last_phase == 1:
            self._debugPrint("PRE-GRASP: successful", 'FG_YELLOW_BRIGHT')

        if self.actor.getPhase() == 3 and self.last_phase == 2:
            self._debugPrint("GRASP: successful", 'FG_YELLOW_BRIGHT')
            gripper_state = self.obj_width    # close gripper after grasp phase

        elif self.actor.getPhase() == 0 and self.last_phase == 3:
            self._debugPrint("POST-GRASP: successful", 'FG_YELLOW_BRIGHT')
            gripper_state = 0.08              # open gripper after post-grasp phase

        else:
            gripper_state = 0                 # no operation

        # generate target pose
        current_pose = transform(self.panda_to_gym, self.current_pose)
        target_pose = transform(current_pose, action[:7])

        self._debugPrint("[panda] Current pose: {}".format(current_pose.tolist()), 'FG_WHITE')
        self._debugPrint("[final] Action: {}".format(action.tolist()), 'FG_WHITE')
        self._debugPrint("[panda] Target pose: {}\n".format(target_pose.tolist()), 'FG_WHITE')

        self.last_phase = self.actor.getPhase()         # update last_phase

        return (target_pose.tolist(), gripper_state)
    

    def _getAction(self):
        # get fingers
        finger0 = finger1 = self.current_gripper / 2.0

        # generate obs for AI
        obs = np.zeros(25)
        obs[:3]    = self.current_pose[:3]     # grip_pos
        obs[3:6]   = self.objOnStart_pose[:3]  # object_pos
        obs[6:9]   = 0                         # object_rel_pos
        obs[9:11]  = [finger0, finger1]        # gripper_state
        obs[11:14] = 0                         # object_rot
        obs[14:17] = 0                         # object_velp
        obs[17:20] = 0                         # object_velr
        obs[20:23] = 0                         # grip_velp
        obs[23:25] = 0                         # gripper_vel

        # adjust object_pos info for retract-ai (It use object_pos as current_pose info)
        if self.actor.getPhase() == 3:
            obs[3:6] = self.current_pose[:3]   # object_pos

        # adjust obs for E2EActor
        if type(self.actor) == E2EActor:
            obs = self.gym_obs

        # get action
        return self.actor.getAction(obs, self.current_pose, self.current_gripper)


   
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
            goal_pose  = transform(self.panda_to_gym, self.goal_pose)
            end_pose = transform(self.panda_to_gym, self.current_pose)
            
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



def main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, FILE_PATH, DEBUG_ENV_ENABLED, ACTOR):
    # for writing
    trajectory = list()

    # initialize Actor
    my_actor = GymEnvironment(DEBUG_ENV_ENABLED, ACTOR)

    # statistics
    results = {
        'goalsAchived': 0,
        'gym_successes': 0,
        'position_errors': list(),
        'orientation_errors': list(),
        'steps': list()
    }

    # start world
    for episode in range(NUM_EPISODES):
        # reset actory
        (target_pose, gripper_state) = my_actor.reset()

        # reset trajectory
        trajectory.clear()

        # add start pose
        trajectory.append(gripper_state)
        trajectory.append(target_pose)
        
        # start episode
        for timer in range(LEN_EPISODE):
            # generate a new action from observations and create a target pose with it
            (target_pose, gripper_state)  = my_actor.getTargetInfo()

            # add target to trajectory and gripper state
            if gripper_state != 0:
                trajectory.append(gripper_state)
            trajectory.append(target_pose)

            # perform a step and get new observations
            my_actor.step()

            # check goal
            (goal_achived, stats) = my_actor.checkGoal()

            # check the output condition
            if goal_achived:
                results['goalsAchived'] += 1
                results['gym_successes'] += stats['gym_success']
                results['position_errors'].append(stats['position_error'])
                results['orientation_errors'].append(stats['orientation_error'])
                results['steps'].append(timer + 1)
                break

        if goal_achived:
            print_col("[Episode {}] Goal achived in {} steps".format(episode, timer + 1), 'FG_GREEN_BRIGHT')
        else:
            print_col("[Episode {}] Goal not achived in {} steps".format(episode, timer + 1), 'FG_RED_BRIGHT')
        
        # write to file
        if WRITE_ENABLE and input("Write to file? [y/n] ") == 'y':
            with open(FILE_PATH, 'w') as file_writer:
                for info in trajectory:
                    # target pose case
                    if type(info) is list:
                        for e in info:
                            file_writer.write("{:>25}  ".format(e))
                        file_writer.write("\n")

                    # gripper state change case
                    else:
                        file_writer.write("{}\n".format(info))
    

    # Generate final statistics
    print("-----------------------------------")
    print_col("All Episodes finish", 'FG_GREEN')

    successes = results['goalsAchived']
    # successes = results['gym_successes']
    fails = NUM_EPISODES - successes
    print_col("accuracy: {}%".format(successes / float(NUM_EPISODES) * 100.0), 'FG_YELLOW_BRIGHT')
    print_col("  - episodes:  {}".format(colorize(str(NUM_EPISODES), 'FG_WHITE')),        'FG_WHITE')
    print_col("  - successes: {}".format(colorize(str(successes),    'FG_GREEN_BRIGHT')), 'FG_WHITE')
    print_col("  - fails:     {}".format(colorize(str(fails),        'FG_RED_BRIGHT')),   'FG_WHITE')
    
    if successes > 0:
        print_col("Mean position errors:    {}"  .format(np.mean(results['position_errors'])), 'FG_YELLOW_BRIGHT')
        print_col("Mean orientation errors: {}"  .format(np.mean(results['orientation_errors'])), 'FG_YELLOW_BRIGHT')
        print_col("Mean steps:              {}\n".format(np.mean(results['steps'])), 'FG_YELLOW_BRIGHT')



if __name__ == "__main__":
    DEBUG_ENV_ENABLED = True
    DEBUG_AI_ENABLED = False
    NUM_EPISODES = 10
    LEN_EPISODE = 150
    WRITE_ENABLE = False
    # FILE_NAME = "trajectory_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    FILE_NAME = "trajectory_test"

    # ACTOR = ACTOR=AiActor(DEBUG_ENABLED=DEBUG_AI_ENABLED)
    ACTOR = ACTOR=E2EActor(DEBUG_ENABLED=DEBUG_AI_ENABLED)
    # ACTOR = ACTOR=HandEngActor(DEBUG_ENABLED=DEBUG_AI_ENABLED)
           

    file_path = os.path.join(os.path.dirname(__file__), "../data/trajectories/" + FILE_NAME + ".txt")
    main(NUM_EPISODES, LEN_EPISODE, WRITE_ENABLE, file_path, DEBUG_ENV_ENABLED, ACTOR)