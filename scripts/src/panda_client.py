#!/usr/bin/env python2

from .panda_msgs import createPandaMsg, processMsg, createErrorMsg
import socket



class GymInterface():
    """Interface to communicate with gym"""

    def __init__(self, HOST, PORT):
        """Open client socket and moveitInterface"""
        # Print client info
        print("Selected: -> {}:{}".format(HOST, PORT))

        # Create a TCP socket at client side
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("TCP client ready")
        self.s.connect((HOST, PORT))
        print("TCP client connected to server {}:{}".format(HOST, PORT))
    

    def __del__(self):
        self.s.close()
        print("TCP Client closed")


    def sendCurrentState(self, current):
        """Send to client the current joints/pose"""
        msg = createPandaMsg(current)
        self.s.send(msg)
    

    def sendError(self):
        """Send to client the error message"""
        msg = createErrorMsg()
        self.s.send(msg)
    

    def getGoalState(self):
        """Wait for message and return goal joints/pose from it or CLOSE"""
        goal = processMsg(self.s.recv(1024))
        return goal




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
    interface = GymInterface(HOST, PORT)


    while True:
        # get current joints/pose and send it
        current = [0, 1, 2, 3, 4, 5, 6, 7]
        print("current: ", current)
        interface.sendCurrentState(current)

        # Wait for goal joints/pose and get it
        goal = interface.getGoalState()
        print("goal: ", goal)

        # Check close
        if goal == 'close':
            break

        # Run goal joints/panda 
        err = raw_input("Error? [y/n] ")
        print(err)
        if  err == 'y':
            interface.sendError()
            break