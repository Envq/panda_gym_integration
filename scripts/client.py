#!/usr/bin/env python3

from utils.joints_msg import createCloseMsg, createJointsMsg, getJointsMsg
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("TCP client ready")
        s.connect((HOST, PORT))
        print("TCP client connected to server {}:{}".format(HOST, PORT))

        # Listen for incoming data from User
        while(True):
            # Check if close
            close = input("Press q to close: ")
            if close == 'q':
                print("Client closed")
                s.send(createCloseMsg())
                break

            # Send joints message
            goal_joints = list()
            for i in range(7):
                goal_joints.append(input("Joint{}: ".format(i)))
            s.send(createJointsMsg(goal_joints))

            # Get server response
            current_joints = getJointsMsg(s.recv(1024))
            print(current_joints)



if __name__ == "__main__":
    main()