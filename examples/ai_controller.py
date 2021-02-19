#!/usr/bin/env python3

# Custom
import sys
sys.path.append("../scripts/")
from src.panda_server import PandaInterface
from src.utils import quaternion_multiply, transform
from src.colors import print_col, colorize

# AI
from ai.panda_actors import AiActor

# Other
import numpy as np



class MoveitEnvironment():
    def __init__(self, DEBUG_ENABLED, HOST, PORT, ACTOR):
        # attributes
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym] trasformation
        self.debug_enabled = DEBUG_ENABLED
        self.obj_width = 0.04                    # [m]
        # panda_gym internally applies this adjustment to actions (in _set_action()), 
        # so you need to apply it here as well 
        self.panda_gym_action_correction = 0.05  # (limit maximum change in position)

        # create panda interface
        self.panda = PandaInterface(HOST, PORT)
        self.panda.getCurrentState() # get panda-client connection

        # create actor
        self.actor = ACTOR
        self.actor.setMaxEpisodeSteps(50)
                    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)


    def reset(self):
        # get object pose on start
        self.objOnStart_pose = np.array([0.7019080739083265, -0.11301889621397332, 0.125, 0.0, 0.0, 0.0, 1.0])
        
        # get goal pose
        self.goal_pose = np.array([0.738619682797228, 0.04043141396766836, 0.5272451383552441, 0.0, 0.0, 0.0, 1.0])
        
        # start msg
        # start_pose = [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916, -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569]
        start_pose = [0.3, 0.0, 0.6,  0.0, 0.0, 0.0, 1.0]
        start_gripper = 0.08
        start_grasp = 0

        # send start state msg
        self.panda.sendGoalState(start_pose + [start_gripper, start_grasp])
        
        # generate pre_grasp pose
        self.preGrasp_pose = self.objOnStart_pose.copy()
        self.preGrasp_pose[2] += 0.031  # [m] above the obj

        # reset attributes
        self.actor.reset(self.goal_pose, self.objOnStart_pose, self.preGrasp_pose)

        # get observation
        self._getObs()

    
    def _getObs(self):
        # get current state msg
        current_msg = self.panda.getCurrentState()

        # Catch real-panda errors
        if current_msg == "error":
            print_col("Abort", 'FG_RED')
            sys.exit()

        # get current tcp pose (on panda_base frame)            
        self.current_pose = np.array(current_msg[:7])       # grip_pos

        # get current fingers width
        self.current_gripper = current_msg[7]               # gripper_state
        finger0 = finger1 = self.current_gripper / 2.0

        # generate obs for AI
        self.obs = np.zeros(25)
        self.obs[:3]    = self.current_pose[:3]     # grip_pos
        self.obs[3:6]   = self.objOnStart_pose[:3]  # object_pos
        self.obs[6:9]   = 0                         # object_rel_pos
        self.obs[9:11]  = [finger0, finger1]        # gripper_state
        self.obs[11:14] = 0                         # object_rot
        self.obs[14:17] = 0                         # object_velp
        self.obs[17:20] = 0                         # object_velr
        self.obs[20:23] = 0                         # grip_velp
        self.obs[23:25] = 0                         # gripper_vel

        # adjust object_pos info for retract-ai (It use object_pos as current_pose info)
        if self.actor.getPhase() == 3:
            self.obs[3:6] = self.current_pose[:3]   # object_pos


    def getTargetInfo(self):
        # get action
        self.action = self.actor.getAction(self.obs, self.current_pose, self.current_gripper)

        # process action
        action = self.action.copy()
        action[:3] *= self.panda_gym_action_correction

        # generate target pose
        self.target_pose = transform(self.current_pose, action[:7])
        self.target_pose[3:] = [0, 0, 0, 1]

        # generate target gripper
        self.target_gripper = self.obj_width if action[7] < 0 else 0.08

        # generate target grasp
        self.target_grasp = 1 if self.action[7] < 0 else 0

        self._debugPrint("Action: {}".format(action.tolist()), 'FG_WHITE')
        self._debugPrint("Current pose: {}".format(self.current_pose.tolist()), 'FG_WHITE')
        self._debugPrint("Target pose: {}\n".format(self.target_pose.tolist()), 'FG_WHITE')


    def step(self):
        # send goal
        self.panda.sendGoalState(self.target_pose.tolist() + [self.target_gripper, self.target_grasp])

        # get observation 
        self._getObs()


    def checkGoal(self):
        if self.actor.goalIsAchieved():
            # get statistics            
            stats = dict()
            stats['position_error'] = np.linalg.norm(self.goal_pose[:3] - self.current_pose[:3])
            stats['orientation_error'] = np.linalg.norm(self.goal_pose_pose[3:] - self.current_pose[3:])
            stats['gym_success'] = self.info['is_success']
            return (True, stats)
        else:
            return (False, None)


    def __del__(self):
        self._debugPrint("close communication", 'FG_MAGENTA')
        self.panda.sendClose()



def main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENABLED, HOST, PORT):
    # initialize Actor
    my_actor = MoveitEnvironment(DEBUG_ENABLED, HOST, PORT, ACTOR=AiActor())

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
        my_actor.reset()
        
        # start episode
        for timer in range(LEN_EPISODE):
            # generate a new action from observations and create a target pose with it
            my_actor.getTargetInfo()

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
    # PARAMETERS
    HOST = "127.0.0.1"
    PORT = 2000
    DEBUG_ENABLED = True
    NUM_EPISODES = 2
    LEN_EPISODE = 150           


    main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENABLED, HOST, PORT)