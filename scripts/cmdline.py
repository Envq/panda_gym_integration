#!/usr/bin/env python2

from src.panda_moveit import MoveGroupInterface
import sys, rospy, moveit_commander


def help():
    print("Commands available:")
    print("p 'px' 'py' 'pz' 'ow' 'ox' 'oy' 'oz' 'fingersDistance'")
    print("j 'j0' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6' 'fingerJoints'")
    print("get pose")
    print("get joints")
    print("quit")
    print("help")



def main(HOST, PORT):
    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_interface', anonymous=True)

        # Create panda moveit interface
        panda = MoveGroupInterface()
        
        help()
        while True:
            # Read command
            command = raw_input("> ")

            if (command == "quit"):
                break
            
            elif (command == "help"):
                help()
            
            elif (command == "get pose"):
                print(panda.getPose())
            
            elif (command == "get joints"):
                print(panda.getJoints())
            
            else:
                cmd = command.split(" ")
                print(cmd)
                if len(cmd) != 9:
                    print("Command not valid")

                else:
                    goal = list()
                    for i in range(8):
                        goal.append(float(cmd[i + 1]))
                    print(goal)

                    if cmd[0] == 'p':
                        print("Success? ", panda.moveToPose(goal))

                    elif cmd[0] == 'j':
                        print("Success? ", panda.moveToJoints(goal))
                
                    else:
                        print("Command not found")

    except rospy.ROSInterruptException:
        print("ROS interrupted")



if __name__ == "__main__":
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    main(HOST, PORT)