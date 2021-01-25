# PANDA GYM MOVEIT
This package is useful for connect [panda_gym](https://github.com/qgallouedec/panda-gym) with ROS for control real Panda robot


## Table of Contents

* [Getting started](#getting-started)
  * [Configuration](#configuration)
* [Nodes](#nodes)
  * [controller](#controller)
  * [simulation](#simulation)
  * [panda_interface](#panda_interface)
* [VSCode](#vscode)
* [Author](#author)
* [License](#license)

---
## Getting started
This package was tested in:
- ROS melodic running under Ubuntu 18.04 in simulation
- ROS kinetic running under Ubuntu 16.04 with real panda arm


## Configuration
**Note:** this setup is for the simulation on your computer. This package is compatible with kinetic environment of the laboratory

Install [ROS](http://wiki.ros.org/melodic/Installation/Ubuntu) and catkin
~~~
sudo apt install cmake ros-melodic-catkin python-catkin-tools
~~~

Install project dependencies
~~~
sudo apt install cmake git
~~~

Prepare workspace
~~~
mkdir ~/panda_gym_ws
cd ~/panda_gym_ws/
mkdir src lib
~~~

Uninstall existing installation of libfranka and franka_ros to avoid conflicts
~~~
sudo apt remove "*libfranka*" "*franka-ros*"
~~~

Get and Build libfranka 0.7.1
~~~
sudo apt install build-essential libpoco-dev libeigen3-dev
cd ~/panda_gym_ws/lib/
git clone --recursive https://github.com/frankaemika/libfranka
cd libfranka/
git checkout 0.7.1
git submodule update
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
~~~

Get franka_ros
~~~
cd ~/panda_gym_ws/src/
git clone --recursive https://github.com/frankaemika/franka_ros
cd franka_ros/
git checkout f7f00b6d9678e59e6f34ccc8d7aad6491b42ac80
~~~

Get panda_moveit_config Moveit
~~~
sudo apt install ros-melodic-moveit
cd ~/panda_gym_ws/src
git clone https://github.com/ros-planning/panda_moveit_config.git
cd panda_moveit_config/
git checkout 5c97a61e9e8a02ca7f1fe08df48ac4ff1b03871a
~~~

Get panda_gym_moveit
~~~
cd ~/panda_gym_ws/src
git clone https://github.com/Envq/panda_gym_moveit.git
~~~

Configure and Build workspace
~~~
echo '#ROS CONFIGS' >> ~/.bashrc
# If you use a different linux distribution based on bionic like Linux Mint
echo 'export ROS_OS_OVERRIDE=ubuntu:18.04:bionic' >> ~/.bashrc

# To fix a bug with moveit
echo 'export LC_NUMERIC="en_US.UTF-8"' >> ~/.bashrc

echo 'source ~/panda_ws/devel/setup.bash' >> ~/.bashrc
source ~/.bashrc

cd ~/panda_ws/
rosdep install -y --from-paths src --ignore-src --rosdistro melodic --skip-keys libfranka

catkin config -j$(nproc) --extend /opt/ros/melodic --cmake-args -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=~/panda_gym_ws/lib/libfranka/build

catkin build
~~~

---
## Nodes
A brief description of the launch files available:

### **simulation**
This node is for testing your scripts on your computer

### **controller**
This node is for running your scripts on laboratory computer connect with Panda

### **panda_interface**
1. Use template to create your code.
2. Run: python3 template
3. Run: roslaunch panda_gym_moveit panda_simulation.launch


![Sequence diagram Close](doc/images/sequence_close.png?raw=true "Sequence diagram Close")

![Sequence diagram Error](doc/images/sequence_error.png?raw=true "Sequence diagram Error")


---
## VSCode
I used these extensions:
- **python** by microsoft
- **ros** by microsoft
- **urdf** by smilerobotics
- **git graph** by mhutchie
- **gruvbox mirror** by adamsome
- **vscode-icons** by icons for visual studio code


---
## Author
**Enrico Sgarbanti** [@**Envq**](https://github.com/Envq)


## License
This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details