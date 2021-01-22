#!/usr/bin/env python2

from utils.joints_msg import createJointsMsg, getJointsMsg
import socket, sys


def getCurrentJoints():
    return (0, 0, 0, 0, 0, 0, 0)


def moveJointsTo(goal_joints):
    return


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

    # Listen for incoming message for User
    while(True):
        # Send current joints message
        s.send(createJointsMsg(getCurrentJoints()))

        # Get server response
        goal_joints = getJointsMsg(s.recv(1024))

        # Check if close
        if goal_joints == 'close':
            print("Client closed")
            break

        # Perform goal joints
        moveJointsTo(goal_joints)
        print("Goal Joints: ", goal_joints)
    s.close()



if __name__ == "__main__":
    main()