#!/usr/bin/env python3

from .utils.joints_msg import createJointsMsg, createCloseMsg, processMsg
from math import pi
import socket



class PandaInterface():
    """Interface to communicate with panda"""

    def __init__(self, HOST, PORT):
        """Open server socket and wait client"""
        print("Selected: -> {}:{}".format(HOST,PORT))
        
        # Create a TCP socket at server side
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # for ignore ack and close socket
        self.s.bind((HOST, PORT))
        self.s.listen() # enable to accept connection
        print("TCP server ready")

        # Wait client connection
        (self.conn, client) = self.s.accept()
        print("TCP server connected to client {}:{}".format(client[0], client[1]))
    

    def __del__(self):
        self.s.close()
        self.conn.close()
        print("TCP Server closed")
        

    def getCurrentJoints(self):
        """Wait for message and return current joints from it or ERROR"""
        return processMsg(self.conn.recv(1024))


    def sendClose(self):
        """Respond to client with close message"""
        self.conn.send(createCloseMsg())
    

    def sendGoalJoints(self, goal_joints):
        """Respond to client with goal joints"""
        self.conn.send(createJointsMsg(goal_joints))



if __name__ == "__main__":
    """TEST"""
    import sys

    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    # Update HOST and PORT
    if (len(sys.argv) == 3):
        HOST = sys.argv[1]
        PORT = int(sys.argv[2])

    # Create gym interface for connect to panda
    interface = PandaInterface(HOST, PORT)

    while True:
        # Get current joints
        current_joints = interface.getCurrentJoints()

        # Check error
        if current_joints == 'error':
            print("Panda Error!")
            break
        else:
            print("Current Joints: {}".format(current_joints))

        # Check close goal
        code = input("Insert Code [q to quit]: ")
        if code == 'q':
            interface.sendClose()
            break
        
        # Process goal joints            
        if code == '1':
            goal_joints = (-1.688, -0.369, 2.081, -2.628, -2.341, 0.454, 0.323)
        elif code == '2':
            goal_joints = (0.00, -0.25 * pi, 0.00, -0.75 * pi, 0.00, 0.50 * pi, 0.25 * pi)
        elif code == '3':
            goal_joints = (0, -pi/4, 0, -pi/2, 0, pi/3, 0)
        else:
            goal_joints = (0, 0, 0, 0, 0, 0, 0)

        # Send goal joints
        print("goal_joints: ", goal_joints)
        interface.sendGoalJoints(goal_joints)