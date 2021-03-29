#!/usr/bin/env python3

# Frankx
from frankx import LinearMotion, Affine, Robot

# Other
import numpy as np
import sys, os

# panda_controller
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../panda_controller/scripts/src")))
from utils import transform, transform_inverse
from colors import print_col, colorize

# Custom
from panda_actors import AiActor, HandEngActor



class FrankxEnvironment():
    def __init__(self, DEBUG_ENABLED, ACTOR, IP, DYNAMIC_REL):
        # attributes
        self.debug_enabled = DEBUG_ENABLED
        # panda_gym internally applies this adjustment to actions (in _set_action()), 
        # so you need to apply it here as well 
        self.panda_gym_action_correction = 0.05  # (limit maximum change in position)
        self.last_target_gripper = 1

        # transformations
        self.panda_to_gym = np.array([-0.6919, -0.7441, -0.3,  0, 0, 0, 1]) # [panda -> gym]
        self.gym_to_panda = transform_inverse(self.panda_to_gym)            # [gym -> panda]

        # create panda interface
        self.arm = Robot(IP)
        self.gripper = self.arm.get_gripper()
        self.arm.set_default_behavior()
        # self.arm.recover_from_errors()

        # Reduce the acceleration and velocity dynamic
        self.arm.set_dynamic_rel(DYNAMIC_REL)

        # create AI actor
        self.actor = ACTOR
                    
    
    def _debugPrint(self, msg, color='FG_DEFAULT'):
        if self.debug_enabled: 
            print_col(msg, color)
    

    def _moveArm(self, pose):
        position = pose[:3]
        pose = Affine(position[0], position[1], position[2])
        motion = LinearMotion(pose)
        self.arm.move(motion)


    def reset(self, START_POSE, OBJ_POSE, GOAL_POSE):
        # get object pose on start
        self.objOnStart_pose = np.array(OBJ_POSE)
        self.gym_to_objOnStart = transform(self.gym_to_panda, self.objOnStart_pose)
        
        # get goal pose
        self.goal_pose = np.array(GOAL_POSE)
        gym_to_goal = transform(self.gym_to_panda, self.goal_pose)
        
        # start msg
        start_pose = START_POSE
        start_gripper = 0.08
        start_grasp = 0

        # go to start pose
        self._moveArm(start_pose)
        self.gripper.move(self.gripper.max_width)
                
        # generate pre_grasp pose
        self.preGrasp_pose = self.objOnStart_pose.copy()
        self.preGrasp_pose[2] += 0.031  # [m] above the obj
        gym_to_preGrasp = transform(self.gym_to_panda, self.preGrasp_pose)

        # reset attributes
        # self.actor.reset(self.goal_pose, self.objOnStart_pose, self.preGrasp_pose)
        self.actor.reset(gym_to_goal, self.gym_to_objOnStart, gym_to_preGrasp)

        # get observation
        self._getObs()

        # debug
        # self._debugPrint("[gym  ] Obj pose:  {}".format(self.gym_to_objOnStart.tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Obj pose:  {}".format(self.objOnStart_pose.tolist()), 'FG_BLUE')
        # self._debugPrint("[gym  ] Goal pose: {}".format(gym_to_goal.tolist()), 'FG_BLUE')
        self._debugPrint("[panda] Goal pose: {}\n".format(self.goal_pose.tolist()), 'FG_BLUE')

    
    def _getObs(self):
        # get current pose
        current = self.arm.current_pose().vector().tolist()

        # get current tcp pose (on panda_base frame)            
        self.current_pose = np.array(current[:3] + [0, 0, 0, 1])       # grip_pos

        # get current fingers width
        self.current_gripper = self.gripper.width()                    # gripper_state


    def getTargetInfo(self):
        # get action
        self.action = self._getAction()

        # process action
        action = self.action.copy()
        action[:3] *= self.panda_gym_action_correction

        # generate target pose
        self.target_pose = transform(self.current_pose, action[:7])
        self.target_pose[3:] = [0, 0, 0, 1]

        # generate target gripper
        self.target_gripper = action[7]

        # debug
        self._debugPrint("[panda] Current pose: {}".format(self.current_pose.tolist()), 'FG_WHITE')
        self._debugPrint("[final] Action: {}".format(action.tolist()), 'FG_WHITE')
        self._debugPrint("[panda] Target pose: {}\n".format(self.target_pose.tolist()), 'FG_WHITE')
    

    def _getAction(self):
        # adjust current pose and get fingers
        gym_to_current = transform(self.gym_to_panda, self.current_pose)
        finger0 = finger1 = self.current_gripper / 2.0

        # self._debugPrint("[gym  ] Current pose: {}".format(gym_to_current.tolist()), 'FG_WHITE')

        # generate obs for AI
        obs = np.zeros(25)
        obs[:3]    = gym_to_current[:3]          # grip_pos
        obs[3:6]   = self.gym_to_objOnStart[:3]  # object_pos
        obs[6:9]   = 0                           # object_rel_pos
        obs[9:11]  = [finger0, finger1]          # gripper_state
        obs[11:14] = 0                           # object_rot
        obs[14:17] = 0                           # object_velp
        obs[17:20] = 0                           # object_velr
        obs[20:23] = 0                           # grip_velp
        obs[23:25] = 0                           # gripper_vel

        # adjust object_pos info for retract-ai (It use object_pos as current_pose info)
        if self.actor.getPhase() == 3:
            obs[3:6] = gym_to_current[:3]        # object_pos

        # get action
        return self.actor.getAction(obs, gym_to_current, self.current_gripper)


    def step(self):
        self._moveArm(self.target_pose)
        if self.target_gripper != self.last_target_gripper:
            if self.target_gripper < 0:
                self.gripper.clamp()
            else:
                self.gripper.move(self.gripper.max_width)
        self.last_target_gripper = self.target_gripper

        # get observation 
        self._getObs()


    def checkGoal(self):
        if self.actor.goalIsAchieved():
            # get statistics            
            stats = dict()
            stats['position_error'] = np.linalg.norm(self.goal_pose[:3] - self.current_pose[:3])
            stats['orientation_error'] = np.linalg.norm(self.goal_pose[3:] - self.current_pose[3:])
            return (True, stats)
        else:
            return (False, None)



def main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENV_ENABLED, ACTOR, IP, DYNAMIC_REL, START_POSE, OBJ_POSE, GOAL_POSE):
    # initialize Actor
    actor_in_env = FrankxEnvironment(DEBUG_ENV_ENABLED, ACTOR, IP, DYNAMIC_REL)

    # statistics
    results = {
        'goalsAchived': 0,
        'position_errors': list(),
        'orientation_errors': list(),
        'steps': list()
    }

    # start world
    for episode in range(NUM_EPISODES):
        # reset actory
        actor_in_env.reset(START_POSE[episode], OBJ_POSE[episode], GOAL_POSE[episode])
                
        # start episode
        for timer in range(LEN_EPISODE):
            # generate a new action from observations and create a target pose with it
            actor_in_env.getTargetInfo()

            # perform a step and get new observations
            actor_in_env.step()

            # check goal
            (goal_achived, stats) = actor_in_env.checkGoal()

            # check the output condition
            if goal_achived:
                results['goalsAchived'] += 1
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
    IP = '192.168.1.2'
    DEBUG_ENV_ENABLED = True
    DEBUG_AI_ENABLED = False
    NUM_EPISODES = 1
    LEN_EPISODE = 150
    DYNAMIC_REL = 0.07


    START_POSE = [ # NOT CHANGE IT...
            [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916,  -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569],
            [0.6014990053878944, 1.5880450818915202e-06, 0.29842061906465916,  -3.8623752044513406e-06, -0.0013073068882995874, -5.91084615330739e-06, 0.9999991454490569],
            [0.4, 0.0, 0.4,  0.0, 0.0, 0.0, 1.0], #run this to see the error
        ]
    OBJ_POSE = [
            [0.618024652107368, -0.018727033651026792, 0.125,  0.0, 0.0, 0.0, 1.0],
            [0.50, 0.0, 0.2,  0.0, 0.0, 0.0, 1.0],
        ]
    GOAL_POSE = [
            [0.4898394521073681, -0.07501663365102673, 0.2,  0.0, 0.0, 0.0, 1.0],
            [0.40, 0.0, 0.4,  0.0, 0.0, 0.0, 1.0],
        ]



    ACTOR = AiActor(DEBUG_ENABLED=DEBUG_AI_ENABLED, MAX_EPISODE_STEPS = 50)
    # ACTOR = HandEngActor(DEBUG_ENABLED=DEBUG_AI_ENABLED, MAX_EPISODE_STEPS = 50)

    main(NUM_EPISODES, LEN_EPISODE, DEBUG_ENV_ENABLED, ACTOR, IP, DYNAMIC_REL, START_POSE, OBJ_POSE, GOAL_POSE)