def createJointsMsg(joints):
    return str.encode(' '.join(map(str, joints)))


def createCloseMsg():
    return str.encode('q')


def getJointsMsg(msg):
    response = msg.decode()
    if response == 'q':
      return 'close'
    joints = list()
    for joint in response.split():
        joints.append(float(joint))
    return joints



if __name__ == '__main__':
    """ TEST """
    joints_msg = createJointsMsg(('1', '1', '1', '1.5', '0', '0', '1'))
    print(joints_msg)
    print(getJointsMsg(joints_msg))

    joints_msg = createJointsMsg((1.3, 1.0, 1, 1, 0, 0, 1))
    print(joints_msg)
    print(getJointsMsg(joints_msg))

    close_msg = createCloseMsg()
    print(close_msg)
    print(getJointsMsg(close_msg))