from frankx import LinearMotion, WaypointMotion, Affine, Waypoint, Robot
from math import pi
import time
import sys, os


def parseLine(line):
    target = list()
    for e in line.split():
        target.append(float(e))
    return (target[:7], target[7])


def planning_waypoints(robot, waypoints):
    print("planning...")
    # Define motion
    motion = WaypointMotion(waypoints, return_when_finished=True)
    # Move robot in sync mode
    print("go!")
    robot.move(motion)


def linear_motion(robot, points):
    print("go!")
    step = 1
    for p in points:
        motion = LinearMotion(p)
        robot.move(motion)
        print("[step {}]: {}".format(step, p))
        step += 1


def async_motion(robot, points, time_delay):
    print("prepare waypoint motion...")
    # Define motion
    motion = WaypointMotion([Waypoint(points[0])], return_when_finished=False)

    # Move robot in async mode
    thread = robot.move_async(motion)

    # Add waypoints runtime
    print("go!")
    step = 1
    for p in points:
        time.sleep(time_delay)
        motion.set_next_waypoint(Waypoint(p))

    input('Press enter to stop\n')
    motion.finish()
    thread.join()


def main(IP, DYNAMIC_REL, FILE_NAME, TIME_DELAY, COMMAND_TYPE):
    # Configuration
    robot = Robot(IP)
    gripper = robot.get_gripper()
    robot.set_default_behavior()
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.set_dynamic_rel(DYNAMIC_REL)

    # Reading file
    FILE_PATH_NAME = os.path.join(os.path.dirname(__file__), "../data/paths/" + FILE_NAME + ".txt")
    file_reader = open(FILE_PATH_NAME, 'r')

    # Get waypoints
    start = True
    points = list()
    waypoints = list()
    for line in file_reader:
        (pose, gripper) = parseLine(line)
        if start:
            start_point = Affine(pose[0], pose[1], pose[2])
            start = False
        else:
            waypoints.append(Waypoint(Affine(pose[0], pose[1], pose[2])))
            points.append(Affine(pose[0], pose[1], pose[2]))
    
    # Go to start
    motion = LinearMotion(start_point)
    robot.move(motion)

    # Start selected motion
    if COMMAND_TYPE == 'planning':
        planning_waypoints(robot, waypoints)
    elif COMMAND_TYPE == 'async':
        async_motion(robot, points, TIME_DELAY)
    elif COMMAND_TYPE == 'linear':
        linear_motion(robot, points)
    else:
        print("command not valid")


if __name__ == '__main__':
    # Configurations
    IP = '192.168.1.2'
    DYNAMIC_REL = 0.08
    FILE_NAME = "path_subtasks"
    TIME_DELAY = 0.1

    # Perform main
    # COMMAND_TYPE = planning, async, linear
    main(IP, DYNAMIC_REL, FILE_NAME, TIME_DELAY, COMMAND_TYPE=sys.argv[1])
