import numpy as np


def quaternion_multiply(quat1, quat2):
    x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    result =  np.array([ x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                         x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
                        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2 ], dtype=np.float64)
    if isinstance(quat1, np.ndarray) or isinstance(quat2, np.ndarray):
        return result
    return result.tolist()



def transform(pose1, pose2):
    position = np.array(pose1[:3]) + np.array(pose2[:3])
    orientation = quaternion_multiply(pose1[3:], pose2[3:])
    result = np.concatenate((position, orientation), axis=None)
    if isinstance(pose1, np.ndarray) or isinstance(pose2, np.ndarray):
        return result
    return result.tolist()



if __name__ == "__main__":
    """TEST with python2"""
    import tf2_ros
    import tf.transformations as t
    
    print("test quaternion_multiply type")
    q = [0.70711, 0, 0, 0.70711]
    print("res: {}".format(quaternion_multiply(q, q)))   # 1 0 0 0

    q = (0.70711, 0, 0, 0.70711)
    print("res: {}".format(quaternion_multiply(q, q)))   # 1 0 0 0

    q = np.array([0.70711, 0, 0, 0.70711])
    print("res: {}\n".format(quaternion_multiply(q, q)))  # 1 0 0 0



    print("test transform type")
    t0 = [0.5, 0.5, 0.5,  0.70711, 0, 0, 0.70711]
    print("res: {}".format(transform(t0, t0)))            # 1 1 1  1 0 0 0

    t0 = (0.5, 0.5, 0.5,  0.70711, 0, 0, 0.70711)
    print("res: {}".format(transform(t0, t0)))            # 1 1 1  1 0 0 0

    t0 = np.array([0.5, 0.5, 0.5,  0.70711, 0, 0, 0.70711])
    print("res: {}\n".format(transform(t0, t0)))          # 1 1 1  1 0 0 0



    print("test quaternion_multiply")
    q1 = np.array([0.70711, 0, 0, 0.70711])
    q2 = np.array([0.70711, 0, 0, 0.70711])
    r1 = quaternion_multiply(q1, q2)   # 1 0 0 0
    print("res1: {}".format(r1))
    r2 = t.quaternion_multiply(q1, q2)   # 1 0 0 0
    print("res2: {}".format(r2))
    print("check: {}\n".format(r1 == r2))

    q1 = np.array([0.70711, 0, 0, 0.70711])
    q2 = np.array([-0.401, 0.660, 0.016, -0.635])
    r1 = quaternion_multiply(q1, q2)
    print("res1: {}".format(r1))
    r2 = t.quaternion_multiply(q1, q2)
    print("res2: {}".format(r2))
    print("check: {}\n".format(r1 == r2))

    for i in range(20):
        q1 = t.random_quaternion()
        q2 = t.random_quaternion()
        r1 = quaternion_multiply(q1, q2)
        print("res1: {}".format(r1))
        r2 = t.quaternion_multiply(q1, q2)
        print("res2: {}".format(r2))
        print("check: {}\n".format(r1 == r2))



    print("test transform")
    p1 = [0.5, 0.5, 0.5]
    o1 = [0.70711, 0, 0, 0.7071]
    p2 = [0.5, 0.5, 0.5,]
    o2 = [0.70711, 0, 0, 0.70711]
    r1p = np.array(p1) + np.array(p2)
    r1o = quaternion_multiply(o1, o2)
    r1 = np.append(r1p, r1o).tolist()
    print("res1: {}".format(r1))
    r2p = np.array(p1) + np.array(p2)
    r2o = t.quaternion_multiply(o1, o2)
    r2 = np.append(r2p, r2o).tolist()
    print("res1: {}".format(r1))
    print("res2: {}".format(r2))
    print("check: {}\n".format(r1 == r2))

    p1 = [0.5, 0.5, 0.5]
    o1 = [0.70711, 0, 0, 0.7071]
    p2 = [0.2, 0.1, 0.2,]
    o2 = [-0.401, 0.660, 0.016, -0.635]
    r1p = np.array(p1) + np.array(p2)
    r1o = quaternion_multiply(o1, o2)
    r1 = np.append(r1p, r1o).tolist()
    print("res1: {}".format(r1))
    r2p = np.array(p1) + np.array(p2)
    r2o = t.quaternion_multiply(o1, o2)
    r2 = np.append(r2p, r2o).tolist()
    print("res1: {}".format(r1))
    print("res2: {}".format(r2))
    print("check: {}\n".format(r1 == r2))

