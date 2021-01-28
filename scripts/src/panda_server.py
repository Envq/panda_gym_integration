#!/usr/bin/env python3

from panda_msgs import createPandaMsg, createCloseMsg, processMsg
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
        

    def getCurrentState(self):
        """Wait for message and return current joints/pose from it or ERROR"""
        current = processMsg(self.conn.recv(1024))
        return current


    def sendClose(self):
        """Respond to client with close message"""
        msg = createCloseMsg()
        self.conn.send(msg)
    

    def sendGoalState(self, goal):
        """Respond to client with goal joints/pose"""
        msg = createPandaMsg(goal)
        self.conn.send(msg)



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
        # Wait for current joints/pose and get it
        current = interface.getCurrentState()
        print("Current State: {}".format(current))

        # Check error
        if current == 'error':
            break

        # Check close goal
        if input("Close? [y/n] ") == 'y':
            interface.sendClose()
            break

        # Process goal and send it
        goal = [10, 11, 12, 13, 14, 15, 16, 17]
        print("goal: ", goal)
        interface.sendGoalState(goal)