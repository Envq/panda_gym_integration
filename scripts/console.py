#!/usr/bin/env python2

from src.panda_moveit import MoveGroupInterface
import sys, rospy, moveit_commander
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def help():
    print("Commands available:")
    print("j 'j0' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6' 'fingerJoints'    -> Joint-Move")
    print("p 'px' 'py' 'pz' 'ox' 'oy' 'oz' 'ow 'fingersDistance'  -> TCP-Pose-Move")
    print("w 'px' 'py' 'pz' 'ox' 'oy' 'oz' 'ow 'fingersDistance'  -> Wrist-Pose-Move")
    print("joints                                                 -> get joints")
    print("pose                                                   -> get pose of TCP")
    print("wrist                                                  -> get wrist pose")
    print("convert 'x' 'y' 'z' 'w'                                -> Quaternion in Euler[Rad]")
    print("convert 'x' 'y' 'z'                                    -> Euler[Rad] in Quaternion")
    print("ready                                                  -> go to ready pose")
    print("custom                                                 -> execute armPose from rosparam")
    print("quit                                                   -> exit from console")
    print("help                                                   -> print this")



def main():
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('console', anonymous=True)

        # Get custom pose from param
        custom_pose = list()
        custom_pose.append(rospy.get_param('~px', 0.0))
        custom_pose.append(rospy.get_param('~py', 0.0))
        custom_pose.append(rospy.get_param('~pz', 0.0))
        custom_pose.append(rospy.get_param('~ox', 0.0))
        custom_pose.append(rospy.get_param('~oy', 0.0))
        custom_pose.append(rospy.get_param('~oz', 0.0))
        custom_pose.append(rospy.get_param('~ow', 1.0))
        # print(custom_pose)

        # Create panda moveit interface
        panda = MoveGroupInterface(1)
        
        help()
        while True:
            # Read command
            command = raw_input("> ")

            if (command == "quit"):
                break

            elif (command == "help"):
                help()
            
            elif (command == "joints"):
                print(panda.getJoints())
            
            elif (command == "pose"):
                print(panda.getPose())

            elif (command == "wrist"):
                print(panda.getPoseWrist())
                        
            elif (command == "ready"):
                 print("Success? ", panda.moveToReady())

            elif (command == "custom"):
                print("Success? ", panda.moveToArmPose(custom_pose))
            
            else:
                cmd = command.split(" ")
                if cmd[0] == 'j':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1]))
                    print("Success? ", panda.moveToJoints(goal))

                elif cmd[0] == 'p':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1]))
                    print("Success? ", panda.moveToPose(goal))

                elif cmd[0] == 'w':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1]))
                    print("Success? ", panda.moveToPoseWrist(goal))
                
                elif cmd[0] == 'convert':
                    val = list()
                    for i in range(len(cmd) - 1):
                        val.append(float(cmd[i + 1]))
                    print(val)
                    if len(val) == 4:
                        print("Euler -> ", euler_from_quaternion(val))

                    elif len(val) == 3:
                        print("Quaternion -> ", quaternion_from_euler(val[0], val[1], val[2]))
                
                    else:
                        print("Command not valid")
                else:
                    print("Command not found")

    except rospy.ROSInterruptException:
        print("ROS interrupted")



if __name__ == "__main__":
    main()