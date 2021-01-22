#!/usr/bin/env python3

import socket, sys
from math import pi
from utils.joints_msg import createJointsMsg, createCloseMsg, getJointsMsg


def processCode(code):
    if code == '1':
        return (0.4, 0.1, 0.4, 1.0, 0, 0, 0)
    elif code == '2':
        return (0, -pi/4, 0, -pi/2, 0, pi/3, 0)
    elif code == '3':
        return (0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi)
    else:
        return (0, 0, 0, 0, 0, 0, 0)


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
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # for ignore ack and close socket
        s.bind((HOST, PORT))
        s.listen() # enable to accept connection
        print("TCP server ready")
        (conn, client) = s.accept()
        clientAddr = client[0]
        clientPort = client[1]
        print("TCP server connected to client {}:{}".format(clientAddr, clientPort))
        print("Press q to close")

        # Listen for incoming message from client
        with conn:
            while True:
                # Get joints message
                msg = conn.recv(1024)
                current_joints = getJointsMsg(msg)
                print("Current Joints: {}".format(current_joints))

                # Process current joints
                code = input("Get code: ")

                # Check if close
                if code == 'q':
                    print("Server closed")
                    conn.send(createCloseMsg())
                    break

                # Create the goal joints and send them
                goal_joints = processCode(code)
                conn.send(createJointsMsg(goal_joints))



if __name__ == "__main__":
    main()