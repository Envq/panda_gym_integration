# **PANDA GYM INTEGRATION**
This package connects [panda_gym](https://github.com/qgallouedec/panda-gym) with the real robot (Panda Franka Emika) through ROS



## **Table of Contents**
* [Getting started](#getting-started)
  * [Dependencies and building](#Dependencies-and-building*)
* [Files](#files)
    * [panda_gym_run](#panda_gym_run)
    * [ai_controller](#ai_controller)
    * [panda_path](#panda_path)
    * [other](#other)
* [VSCode](#vscode)
* [Author](#author)
* [License](#license)



---
## **Getting started**
This package was tested with real robot with:
- Ubuntu 20.04 LTS Focal Fossa
- ROS noetic
- [libfranka](https://github.com/frankaemika/libfranka) 0.8.0
- [franka_ros](https://github.com/frankaemika/franka_ros) 0.7.1
- [panda_moveit_config](https://github.com/ros-planning/panda_moveit_config) 0.7.5
- [panda_controller](https://github.com/Envq/panda_controller) noetic_dev
- [frankx](https://github.com/pantor/frankx) master on **cc81e52045be280ee2f6b928e954d05ccc7798ad**



## **Dependencies and building**
Follow **Dependencies and building** section [here](https://github.com/Envq/panda_controller) 

Install [frankx](https://github.com/pantor/frankx)

### **panda_gym**:
The AI has been trained with the goal closest to the default one... 

Go to panda_gym/envs/assets/pickandplace.json and set the object -> body -> basePosition to [0.025, 0.025, 0.4] for better results.


### **Get panda_gym_integration:**
~~~
cd ~/panda_ws/src/

pip3 install --user numpy panda_gym torch

git clone https://github.com/Envq/panda_gym_integration.git

cd ~/panda_ws

catkin build
~~~



---
## **Files:**

### **panda_gym_run**
- This file allows you to run panda_gym and test your policy defined in src/panda_actors.py

- Parameters available:
    - **DEBUG_ENV_ENABLED:** enable panda_gym environment debug prints
    - **DEBUG_AI_ENABLED:** enable ai actor debug prints
    - **NUM_EPISODES:** number of episodes to run
    - **LEN_EPISODE:** maximum number of steps per episode
    - **WRITE_ENABLE:** if true, it asks you whether to save the path generated by the episode performed
    - **FILE_NAME:** specify here the name of the file containing the path to be saved
    - **ACTOR:** specify here your actor

![gif_panda_gym_run](docs/panda_gym_run.gif)


### **ai_controller**
- This file allows you to use your AI to control the real robot with [frankx](https://github.com/pantor/frankx)

- Parameters available:
    - **IP**: specify here the ip of the robot
    - **DEBUG_ENV_ENABLED**: enable panda_gym environment debug prints
    - **DEBUG_AI_ENABLED**: enable ai actor debug prints
    - **NUM_EPISODES**: number of episodes to run
    - **LEN_EPISODE**: maximum number of steps per episode
    - **DYNAMIC_REL**: if this value is 0.08, it set velocity, acceleration and jerk to 8% of the maximum
    - **ACTOR**: specify here your actor
    - **START_POSE**: this is a list of start pose for your episodes. Note**: the AI used requires that the start pose is always the same
    - **OBJ_POSE**: this is a list of object pose for your episodes
    - **GOAL_POSE**: this is a list of goal pose for your episodes


### **panda_path**
- panda_path_frankx (or panda_path_moveit) is a file that generates a trajectory with the path contained in the specified file and executes it with [frankx](https://github.com/pantor/frankx) (or [moveit](https://moveit.ros.org/))

- Parameters available for **panda_path_frankx**:
    - **IP**: specify here the ip of the robot
    - **DYNAMIC_REL**: if this value is 0.08, it set velocity, acceleration and jerk to 8% of the maximum
    - **FILE_NAME**: is the name of the file in data/paths where read the path
    - **TIME_DELAY**: is the delay after which to add new waypoints in "async" mode

    execute:
    - **python3 panda_path_frankx.py planning** ---> to generate the trajectory offline
    - **python3 panda_path_frankx.py linear**   --------> to move to each point at runtime with linear movements
    - **python3 panda_path_frankx.py planning** ---> to generate the trajectory online receiving waypoints with a TIME_DELAY delay


- Parameters available for **panda_path_moveit** in launch/panda_path.launch** (Note**: execute this file with roslaunch panda_gym_integration panda_path.launch):
    - **arm_speed**: is speed used for setMaxVelocityScalingFactor() method of Moveit
    - **gripper_speed**: is the speed used for the move() method of the gripper
    - **gripper_speed, grasp_speed, grasp_force, grasp_epsilon_inner, grasp_epsilon_outer**: are the values used in grasp() method of the gripper

    - **eef_step, jump_threashould**: are the values used in compute_cartesian_path() method of Moveit

    - **real_robot**: if true, panda_controller will use franka_gripper's action clients
    - **file_name**: is the name of the file in data/paths where read the path

    - **mode = 1**: the generation of the trajectories is done with the compute_cartesian_path() method of Moveit
    - **mode = 2**: each point is read and immediately executed with moveArmPoseTCP() method of panda_controller
    - **mode2_delay, mode2_delay**: If mode is 2, each point is execute with mode2_delay and with go(wait=mode2_wait) in Moveit

![gif_panda_path_moveit](docs/panda_path_moveit.gif)


### **other**
  - **debug_frankx**: use python3 debug_frankx.py cmd_name, where **cmd_name** is the command name you want to test with frankx
  - **src/panda_actors**: this files containes the policies and AI, used in **panda_gym_run** and **ai_controller**
  - **ai/**: this folder contains the necessary files for the AI taken [here] (https://github.com/LM095/Multi-Subtask-DRL-for-Pick-and-Place-Task)



---
## VSCode
I used these extensions:
- **c/c++** by microsoft
- **c++ intellisense** by austin
- **cmake** by twxs
- **doxygen** by christoph schlosser
- **clang-format** by xaver
- **doxgen documentation** by christoph schlosser
- **python** by microsoft
- **git graph** by mhutchie
- **gruvbox mirror** by adamsome
- **vscode-icons** by icons for visual studio code
- **git graph** by mhutchie



---
## Author
**Enrico Sgarbanti** [@**Envq**](https://github.com/Envq) for panda-gym integration with real robot

**Luca Marzari** [@**LM095**](https://github.com/LM095) for AI. look [here](https://github.com/LM095/Multi-Subtask-DRL-for-Pick-and-Place-Task) for more details



## License
This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details