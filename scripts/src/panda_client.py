#!/usr/bin/env python2

from .utils.joints_msg import createJointsMsg, createErrorMsg, processMsg
from .utils.panda_moveit_interface import MoveGroupInterface
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

        # Create MoveGroupInterface
        self.panda = MoveGroupInterface()
    

    def __del__(self):
        self.s.close()
        print("TCP Client closed")
    

    def sendCurrentJoints(self):
        """Send to client the current joints"""
        msg = createJointsMsg(self.panda.getJoints())
        self.s.send(msg)
    

    def sendError(self):
        """Send to client the error message"""
        msg = createErrorMsg()
        self.s.send(msg)
    

    def getGoalJoints(self):
        """Wait for message and return goal joints from it or CLOSE"""
        return processMsg(self.s.recv(1024))

    
    def movePandaTo(self, goal_joints):
        """Perform goal joints and returns if the movement was successful"""
        return self.panda.moveToJoints(goal_joints)



if __name__ == "__main__":
    """TEST"""
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
        interface = GymInterface(HOST, PORT)

        while True:
            # Send current joints
            interface.sendCurrentJoints()

            # Get goal joints
            goal_joints = interface.getGoalJoints()

            # Check close
            if goal_joints == 'close':
                break

            # Run goal joints
            success = interface.movePandaTo(goal_joints)
            print("Success? ", success)
            if not success:
                print("Error! abort")
                interface.sendError()
                break

    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except KeyboardInterrupt:
        print("Keboard quit")