#!/usr/bin/env python2

from .utils.joints_msg import createJointsMsg, getJointsMsg
from .utils.panda_moveit_interface import MoveGroupInterface
import socket


class panda_interface():
    def __init__(self, HOST, PORT):
        """ Open client socket and moveitInterface """
        # Print client info
        print("Selected: -> {}:{}".format(HOST, PORT))

        # Create a TCP socket at client side
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("TCP client ready")
        self.s.connect((HOST, PORT))
        print("TCP client connected to server {}:{}".format(HOST, PORT))

        # Create MoveGroupInterface
        self.panda = MoveGroupInterface()
    

    def __del__(self):
        self.s.close()
        print("TCP Client closed")
    

    def sendCurrentJoints(self):
        """ Send to client the current joints """
        msg = createJointsMsg(self.panda.getJoints())
        self.s.send(msg)
    

    def getResponse(self):
        """ Wait for message and process it """
        goal_joints = getJointsMsg(self.s.recv(1024))

        # Check if close
        if goal_joints == 'close':
            return goal_joints

        # Perform goal joints and return status
        return self.panda.moveToJoints(goal_joints)



if __name__ == "__main__":
    import sys, rospy, moveit_commander

    # Connection config
    HOST = "127.0.0.1"
    PORT = 2000

    try:
        # Initialize moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_interface', anonymous=True)

        # Update HOST and PORT
        if (len(sys.argv) == 3):
            HOST = sys.argv[1]
            PORT = int(sys.argv[2])

        # Create gym interface for connect to panda
        interface = panda_interface(HOST, PORT)

        while True:
            interface.sendCurrentJoints()
            response = interface.getResponse()
            if response == 'close':
                break
            print("Success? ", response)

    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except KeyboardInterrupt:
        print("Keboard quit")