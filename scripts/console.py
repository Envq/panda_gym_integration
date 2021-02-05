#!/usr/bin/env python2

from src.panda_moveit import MoveGroupInterface
import sys, rospy, moveit_commander
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def help():
    print("Commands available:")
    print("Notes:")
    print("  ',' are automatically ignored")
    print("  fingerJoints and fingersDistance are optional")
    print("j 'j0' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6' 'fingerJoints'    -> Joint-Move")
    print("p 'px' 'py' 'pz' 'ox' 'oy' 'oz' 'ow 'fingersDistance'  -> TCP-Pose-Move")
    print("w 'px' 'py' 'pz' 'ox' 'oy' 'oz, 'ow 'fingersDistance'  -> Wrist-Pose-Move")
    print("joints                                                 -> get joints")
    print("pose                                                   -> get TCP pose")
    print("tcp                                                    -> get TCP pose")
    print("wrist                                                  -> get wrist pose")
    print("convert 'x' 'y' 'z' 'w'                                -> Quaternion in Euler[Rad]")
    print("convert 'x' 'y' 'z'                                    -> Euler[Rad] in Quaternion")
    print("ready                                                  -> go to ready pose")
    print("custom1                                                -> execute moveToArmPoseTCP from rosparam")
    print("custom2                                                -> execute moveToPose from rosparam")
    print("quit                                                   -> exit from console")
    print("help                                                   -> print this")



def main():
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('console', anonymous=True)

        # Get custom pose from param
        custom_pose1 = list()
        custom_pose1.append(rospy.get_param('~px1', 0.0))
        custom_pose1.append(rospy.get_param('~py1', 0.0))
        custom_pose1.append(rospy.get_param('~pz1', 0.0))
        custom_pose1.append(rospy.get_param('~ox1', 0.0))
        custom_pose1.append(rospy.get_param('~oy1', 0.0))
        custom_pose1.append(rospy.get_param('~oz1', 0.0))
        custom_pose1.append(rospy.get_param('~ow1', 1.0))
        # print(custom_pose1)

        custom_pose2 = list()
        custom_pose2.append(rospy.get_param('~px2', 0.0))
        custom_pose2.append(rospy.get_param('~py2', 0.0))
        custom_pose2.append(rospy.get_param('~pz2', 0.0))
        custom_pose2.append(rospy.get_param('~ox2', 0.0))
        custom_pose2.append(rospy.get_param('~oy2', 0.0))
        custom_pose2.append(rospy.get_param('~oz2', 0.0))
        custom_pose2.append(rospy.get_param('~ow2', 1.0))
        custom_pose2.append(rospy.get_param('~fd2', 0.0))
        custom_pose2.append(rospy.get_param('~gr2', 1.0))
        # print(custom_pose2)

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
            
            elif (command == "tcp"):
                print(panda.getPoseTCP())

            elif (command == "wrist"):
                print(panda.getPoseWrist())

            elif (command == "pose"):
                print(panda.getPose())
                        
            elif (command == "ready"):
                 print("Success? ", panda.moveToReady())

            elif (command == "custom1"):
                print("Success? ", panda.moveToArmPoseTCP(custom_pose1))

            elif (command == "custom2"):
                print("Success? ", panda.moveToPose(custom_pose2))
            
            else:
                cmd = command.split(" ")
                print(cmd)
                if cmd[0] == 'j':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("Success? ", panda.moveToArmJoints(goal))
                    elif len(goal) == 8:
                        print("Success? ", panda.moveToJoints(goal))
                    else:
                        print("Command not valid")

                elif cmd[0] == 'p':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("Success? ", panda.moveToArmPoseTCP(goal))
                    elif len(goal) == 8:
                        print("Success? ", panda.moveToPoseTCP(goal))
                    else:
                        print("Command not valid")

                elif cmd[0] == 'w':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("Success? ", panda.moveToArmPoseWrist(goal))
                    elif len(goal) == 8:
                        print("Success? ", panda.moveToPoseWrist(goal))
                    else:
                        print("Command not valid")
                
                elif cmd[0] == 'convert':
                    val = list()
                    for i in range(len(cmd) - 1):
                        val.append(float(cmd[i + 1].replace(',','')))
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