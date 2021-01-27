

def createPandaMsg(msg):
    return str.encode(' '.join(map(str, msg)))


def createCloseMsg():
    return str.encode('close')


def createErrorMsg():
    return str.encode('error')


def processMsg(msg):
    response = msg.decode()
    if response == 'close' or response == 'error':
        return response
    msg = list()
    for e in response.split():
        msg.append(float(e))
    return msg



if __name__ == '__main__':
    """TEST"""
    
    print("Create joints message")
    panda_msg = createPandaMsg(('1', '1', '1', '1.5', '0', '0', '1'))
    print(panda_msg)
    print(processMsg(panda_msg))
    print()

    panda_msg = createPandaMsg((1.3, 1.0, 1, 1, 0, 0, 1))
    print(panda_msg)
    print(processMsg(panda_msg))
    print()

    panda_msg = createPandaMsg((1.3, 1.0, 1, 1, 0, 0, 1, 2.0, 3.0))
    print(panda_msg)
    print(processMsg(panda_msg))
    print()

    print("Create close message")
    close_msg = createCloseMsg()
    print(close_msg)
    print(processMsg(close_msg))
    print()

    print("Create error message")
    error_msg = createErrorMsg()
    print(error_msg)
    print(processMsg(error_msg))