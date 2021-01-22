#!/usr/bin/env python3

import socket, sys
from utils.joints_msg import createJointsMsg, createCloseMsg, getJointsMsg


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

    # Create a TCP socket at server side
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen() # enable to accept connection
        print("TCP server ready")
        (conn, client) = s.accept()
        clientAddr = client[0]
        clientPort = client[1]
        print("TCP server connected to client {}:{}".format(clientAddr, clientPort))

        # Listen for incoming message from client
        with conn:
            while True:
                # Get joints message
                msg = conn.recv(1024)
                current_joints = getJointsMsg(msg)

                # Check if close
                close = input("Press q to close: ")
                if close == 'q':
                    print("Server closed")
                    conn.send(createCloseMsg())
                    break

                # Read current joints state
                print("Current Joints: {}".format(current_joints))

                # Process the goal joints and send them
                goal_joints = list()
                for i in range(7):
                    goal_joints.append(input("Joint{}: ".format(i)))
                conn.send(createJointsMsg(goal_joints))



if __name__ == "__main__":
    main()