#!/usr/bin/env python2

import socket, sys
from utils.joints_msg import createJointsMsg, getJointsMsg


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
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)
    print("TCP server ready")
    (conn, client) = s.accept()
    clientAddr = client[0]
    clientPort = client[1]
    print("TCP server connected to client {}:{}".format(clientAddr, clientPort))

    while True:
        # Get joints message
        msg = conn.recv(1024)
        goal_joints = getJointsMsg(msg)

        # Check if close
        if goal_joints == 'close':
            print("Server closed")
            conn.close()
            break

        # Perform action
        print("> {}".format(goal_joints))

        # Respond with current joints
        conn.send(createJointsMsg(goal_joints))
    s.close()



if __name__ == "__main__":
    main()