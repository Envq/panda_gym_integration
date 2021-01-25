

def createJointsMsg(joints):
    return str.encode(' '.join(map(str, joints)))


def createCloseMsg():
    return str.encode('close')


def createErrorMsg():
    return str.encode('error')


def processMsg(msg):
    response = msg.decode()
    if response == 'close' or response == 'error':
        return response
    joints = list()
    for joint in response.split():
        joints.append(float(joint))
    return joints



if __name__ == '__main__':
    """TEST"""
    
    print("Create joints message")
    joints_msg = createJointsMsg(('1', '1', '1', '1.5', '0', '0', '1'))
    print(joints_msg)
    print(processMsg(joints_msg))

    joints_msg = createJointsMsg((1.3, 1.0, 1, 1, 0, 0, 1))
    print(joints_msg)
    print(processMsg(joints_msg))

    print("\nCreate close message")
    close_msg = createCloseMsg()
    print(close_msg)
    print(processMsg(close_msg))

    print("\nCreate error message")
    error_msg = createErrorMsg()
    print(error_msg)
    print(processMsg(error_msg))