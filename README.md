# PANDA GYM INTEGRATION
This package connects [panda_gym](https://github.com/qgallouedec/panda-gym) with the real robot (Panda Franka Emika) through ROS



## Table of Contents
* [Getting started](#getting-started)
  * [Dependencies and building](#Dependencies-and-building*)
* [VSCode](#vscode)
* [Author](#author)
* [License](#license)



---
## Getting started
This package was tested with real robot with:
- Ubuntu 20.04 LTS Focal Fossa
- ROS noetic
- [libfranka](https://github.com/frankaemika/libfranka) 0.8.0
- [franka_ros](https://github.com/frankaemika/franka_ros) 0.7.1
- [panda_moveit_config](https://github.com/ros-planning/panda_moveit_config) 0.7.5
- [panda_controller](https://github.com/Envq/panda_controller) noetic_dev



## Dependencies and building
Follow **Dependencies and building** section [here](https://github.com/Envq/panda_controller) 

**Get panda_gym_integration:**
~~~
cd ~/panda_ws/src/

pip3 install --user numpy panda_gym torch

git clone https://github.com/Envq/panda_gym_integration.git

cd ~/panda_ws

catkin build
~~~



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