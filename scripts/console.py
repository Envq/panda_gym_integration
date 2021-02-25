#!/usr/bin/env python2

from src.panda_moveit import PandaMoveitInterface
import sys, rospy, moveit_commander
from src.colors import colorize, print_col
from src.utils import quaternion_from_euler, euler_from_quaternion


def help():
    color = 'FG_MAGENTA'
    print_col("  Commands available:", color)
    print_col("Note: ',' are automatically ignored\n", color)
    print_col("j 'j0' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6'", color)
    print_col("w 'px' 'py' 'pz'  'ox' 'oy' 'oz, 'ow'", color)
    print_col("t 'px' 'py' 'pz'  'ox' 'oy' 'oz' 'ow'", color)
    print_col("p 'px' 'py' 'pz'  'ox' 'oy' 'oz' 'ow'  'fd' 'grasp'", color)
    print_col("joints", color)
    print_col("wrist", color)
    print_col("tcp", color)
    print_col("pose", color)
    print_col("gripper", color)
    print_col("gripper homing", color)
    print_col("gripper 'width' 'speed'", color)
    print_col("grasp 'width' 'speed' 'force' 'epsilon_inner' 'epsilon_outer'", color)
    print_col("convert 'x' 'y' 'z' 'w'", color)
    print_col("convert 'roll' 'pitch' 'yaw'", color)
    print_col("ready", color)
    print_col("custom", color)
    print_col("quit", color)
    print_col("help\n", color)



def main():
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('console', anonymous=True)

        # Get real_robot
        real_robot = rospy.get_param('~real_robot', False)

        # Get custom pose from param
        custom_pose = list()
        custom_pose.append(rospy.get_param('~px', 0.0))
        custom_pose.append(rospy.get_param('~py', 0.0))
        custom_pose.append(rospy.get_param('~pz', 0.0))
        custom_pose.append(rospy.get_param('~ox', 0.0))
        custom_pose.append(rospy.get_param('~oy', 0.0))
        custom_pose.append(rospy.get_param('~oz', 0.0))
        custom_pose.append(rospy.get_param('~ow', 1.0))


        # Create panda moveit interface
        panda = PandaMoveitInterface(delay=1, real_robot=real_robot)
        
        help()
        while True:
            # Read command
            command = raw_input(colorize("> ", 'FG_GREEN'))

            if (command == "quit"):
                break

            elif (command == "help"):
                help()
            
            elif (command == "joints"):
                print("  {}".format(panda.getArmJoints()))

            elif (command == "wrist"):
                print("  {}".format(panda.getArmPoseWrist()))
            
            elif (command == "tcp"):
                print("  {}".format(panda.getArmPoseTCP()))

            elif (command == "pose"):
                print("  {}".format(panda.getPose()))

            elif (command == "gripper"):
                print("  {}".format(panda.getGripper()))
                        
            elif (command == "ready"):
                 print("  Success? {}".format(panda.moveReady()))

            elif (command == "custom"):
                print("  Success? {}".format(panda.moveArmPoseTCP(custom_pose)))
            
            else:
                cmd = command.split(" ")
                cmd = list(filter(lambda e: e != '', cmd))
                if cmd[0] == 'j':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("  Success? {}".format(panda.moveArmJoints(goal)))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')

                elif cmd[0] == 't':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("  Success? {}".format(panda.moveArmPoseTCP(goal)))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')

                elif cmd[0] == 'w':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 7:
                        print("  Success? {}".format(panda.moveArmPoseWrist(goal)))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')

                elif cmd[0] == 'p':
                    goal = list()
                    for i in range(len(cmd) - 1):
                        goal.append(float(cmd[i + 1].replace(',','')))
                    if len(goal) == 9:
                        print("  Success? {}".format(panda.movePose(goal)))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')
                
                elif cmd[0] == 'convert':
                    val = list()
                    for i in range(len(cmd) - 1):
                        val.append(float(cmd[i + 1].replace(',','')))
                    if len(val) == 4:
                        print("  Euler -> {}".format(euler_from_quaternion(val[0], val[1], val[2], val[3])))
                    elif len(val) == 3:
                        print("  Quaternion -> {}".format(quaternion_from_euler(val[0], val[1], val[2])))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')
                
                elif cmd[0] == 'gripper':
                    val = list()
                    for i in range(len(cmd) - 1):
                        val.append(cmd[i + 1].replace(',',''))
                    if len(val) == 1:
                        if val[0] == 'homing':
                            panda.homingGripper()
                        else:
                            panda.moveGripper(float(val[0]))
                    elif len(val) == 2:
                        panda.moveGripper(float(val[0]), float(val[1]))
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')
                
                elif cmd[0] == 'grasp':
                    val = list()
                    for i in range(len(cmd) - 1):
                        val.append(float(cmd[i + 1].replace(',','')))
                    if len(val) == 1:
                        panda.graspGripper(val[0])
                    elif len(val) == 2:
                        panda.graspGripper(val[0], val[1])
                    elif len(val) == 3:
                        panda.graspGripper(val[0], val[1], val[2])
                    elif len(val) == 5:
                        panda.graspGripper(val[0], val[1], val[2], val[3], val[4])
                    else:
                        print_col("  Command not valid", 'FG_YELLOW')

                else:
                    print_col("  Command not found", 'FG_YELLOW')

    except rospy.ROSInterruptException:
        print_col("ROS interrupted", 'FG_RED')



if __name__ == "__main__":
    main()