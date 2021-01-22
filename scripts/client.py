#!/usr/bin/env python2

from utils.joints_msg import createJointsMsg, getJointsMsg
from utils.panda_interface import MoveGroupInterface
import sys, rospy, moveit_commander
import socket, sys


def main():
    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Update HOST
    if (len(sys.argv) == 2):
        HOST = sys.argv[1]
    if (len(sys.argv) == 3):
        HOST = sys.argv[1]
        PORT = int(sys.argv[2])
    print("Selected: -> {}:{}".format(HOST,PORT))

    # Create a TCP socket at client side
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("TCP client ready")
    s.connect((HOST, PORT))
    print("TCP client connected to server {}:{}".format(HOST, PORT))

    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_interface_node', anonymous=True)

    # Create MoveGroupInterface
    panda = MoveGroupInterface()

    # Listen for incoming message for User
    while(True):
        # Send current joints message
        s.send(createJointsMsg(panda.getJoints()))

        # Get server response
        goal_joints = getJointsMsg(s.recv(1024))

        # Check if close
        if goal_joints == 'close':
            print("Client closed")
            break

        # Perform goal joints
        success = panda.moveToJoints(goal_joints)
        print("Success: ", success)
    s.close()



if __name__ == "__main__":
    main()