import numpy as np
import math



def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler [roll(x), pitch(y), raw(z)] in Quaternion [x, y, z, w]"""
    return np.array([ np.sin(roll*0.5) * np.cos(pitch*0.5) * np.cos(yaw*0.5) - np.cos(roll*0.5) * np.sin(pitch*0.5) * np.sin(yaw*0.5),
                      np.cos(roll*0.5) * np.sin(pitch*0.5) * np.cos(yaw*0.5) + np.sin(roll*0.5) * np.cos(pitch*0.5) * np.sin(yaw*0.5),
                      np.cos(roll*0.5) * np.cos(pitch*0.5) * np.sin(yaw*0.5) - np.sin(roll*0.5) * np.sin(pitch*0.5) * np.cos(yaw*0.5),
                      np.cos(roll*0.5) * np.cos(pitch*0.5) * np.cos(yaw*0.5) + np.sin(roll*0.5) * np.sin(pitch*0.5) * np.sin(yaw*0.5) ],
                    dtype=np.float64)


def euler_from_quaternion(x, y, z, w):
    """Convert Quaternion [x, y, z, w] in Euler [roll(x), pitch(y), raw(z)]"""
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if (np.abs(sinp) >= 1):
        pitch = np.copysign(np.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quaternion_multiply(quaternion1, quaternion2):
    """Quaternion [x, y, z, w] multiply"""
    x1, y1, z1, w1 = quaternion1
    x2, y2, z2, w2 = quaternion2
    return np.array([ x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                       -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
                       -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2 ],
                    dtype=np.float64)


def quaternion_inverse(quaternion):
    """Quaternion [x, y, z, w] inverse"""
    x, y, z, w = quaternion
    return np.array((-x, -y, -z, w), dtype=np.float64)


def rotation_matrix(quaternion):
    """Get rotation matrix from Quaternion [x, y, z, w] """
    x, y, z, w = quaternion
    return np.array([ [2*(w*w+x*x)-1,  2*(x*y-w*z),     2*(x*z+w*y)   ],
                      [2*(x*y+w*z),    2*(w*w+y*y)-1,   2*(y*z-w*x)   ],
                      [2*(x*z-w*y),    2*(y*z+w*x),     2*(w*w+z*z)-1 ]],
                    dtype=np.float64)


def quaternion_from_matrix(matrix):
    w = 0.5 * np.sqrt(matrix[0,0] + matrix[1,1] + matrix[2,2] + 1)
    x = 0.5 * np.sign(matrix[2,1] - matrix[1,2]) * np.sqrt(matrix[0,0] - matrix[1,1] - matrix[2,2] + 1)
    y = 0.5 * np.sign(matrix[0,2] - matrix[2,0]) * np.sqrt(matrix[1,1] - matrix[2,2] - matrix[0,0] + 1)
    z = 0.5 * np.sign(matrix[1,0] - matrix[0,1]) * np.sqrt(matrix[2,2] - matrix[0,0] - matrix[1,1] + 1)
    return np.array([x, y, z, w])


def homogeneous_matrix(translation, rotation):
    """Get homogeneous matrix from Rotation [x, y, z, w] and Translation [x, y, z]"""
    translation  = np.array(translation).reshape(3,1)
    rotation = rotation_matrix(rotation)
    padding = np.array([0, 0, 0, 1]).reshape(1,4)
    return np.append(np.append(rotation, translation, axis=1), padding, axis=0)


def homogeneous_matrix_inverse(matrix):
    """Homogeneous matrix [Rotation [x, y, z, w], Translation [x, y, z]] inverse"""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3:4]
    padding = matrix[3:4, :]
    inverted_rotation = rotation.transpose()
    new_translation = np.dot(-inverted_rotation, translation)
    return np.append(np.append(inverted_rotation, new_translation, axis=1), padding, axis=0)


def transform_from_matrix(matrix):
    """Get transform [px, py, pz,  qx, qy, qz, qw] from homogeneous matrix"""
    rot_matrix = matrix[:3, :3]
    translation = matrix[:3, 3:4].ravel()
    rotation = quaternion_from_matrix(rot_matrix).ravel()
    return np.concatenate((translation, rotation), axis=None)


def transform_inverse(pose):
    matrix = homogeneous_matrix(pose[:3], pose[3:])
    matrix_inv = homogeneous_matrix_inverse(matrix)
    return transform_from_matrix(matrix_inv)


def transform(pose1, pose2):
    matrix1 = homogeneous_matrix(pose1[:3], pose1[3:])
    matrix2 = homogeneous_matrix(pose2[:3], pose2[3:])
    matrix3 = np.dot(matrix1, matrix2)
    return transform_from_matrix(matrix3)





if __name__ == "__main__":
    """TEST with python2"""
    import tf.transformations as t

    test_enabled = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    num_test = 50
    tot_test = 0
    tot_successess = 0

    # quaternion_from_euler and euler_from_quaternion
    if test_enabled[0]:
        print("Test Conversions:")
        quaternion_successes = 0
        euler_successes = 0
        for i in range(num_test):
            # generate random
            q = t.random_quaternion()
            # test euler_from_quaternion
            e1 = t.euler_from_quaternion(q)
            e2 = euler_from_quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            quaternion_successes += int(np.allclose(e1, e2))
            # test quaternion_from_euler
            q1 = t.quaternion_from_euler(e1[0], e1[1], e1[2])
            q2 = quaternion_from_euler(roll=e2[0], pitch=e2[1], yaw=e2[2])
            euler_successes += int(np.allclose(q1, q2))
        print('   -> Quaternion successes: {}/{}'.format(quaternion_successes, num_test))
        print('   -> Euler successes: {}/{}'.format(euler_successes, num_test))
        print("--------------------\n")
        tot_successess += quaternion_successes + euler_successes
        tot_test += num_test * 2


    # quaternion_multiply
    if test_enabled[1]:
        print("test quaternion_multiply")
        successes = 0
        for i in range(num_test):
            q1 = t.random_quaternion()
            q2 = t.random_quaternion()
            r1 = t.quaternion_multiply(q1, q2)
            r2 = quaternion_multiply(q1, q2)
            successes += int(np.allclose(r1, r2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test


    # quaternion_inverse
    if test_enabled[2]:
        print("test quaternion_inverse")
        successes = 0
        for i in range(num_test):
            q = t.random_quaternion()
            q1 = t.quaternion_inverse(q)
            q2 = quaternion_inverse(q)
            successes += int(np.allclose(q1, q2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test


    # rotation_matrix
    if test_enabled[3]:
        print("test rotation_matrix")
        successes = 0
        for i in range(num_test):
            q = t.random_quaternion()
            m1 = t.quaternion_matrix(q)[:3, :3]
            m2 = rotation_matrix(q)
            successes += int(np.allclose(m1, m2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test
    

    # quaternion_from_matrix
    if test_enabled[4]:
        print("test quaternion_from_matrix")
        successes = 0
        for i in range(num_test):
            m = t.random_rotation_matrix()
            q1 = t.quaternion_from_matrix(m)
            q2 = quaternion_from_matrix(m[:3, :3])
            successes += int(np.allclose(q1, q2) or np.allclose(q1, -q2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test
    
    
    # homogeneous_matrix
    if test_enabled[5]:
        print("test homogeneous_matrix")
        successes = 0
        for i in range(num_test):
            q = t.random_quaternion()
            p = [0, 0, 0]
            m1 = t.quaternion_matrix(q)
            m2 = homogeneous_matrix(p, q)
            successes += int(np.allclose(m1, m2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test


    # homogeneous_matrix_inverse
    if test_enabled[6]:
        print("test homogeneous_matrix_inverse")
        successes = 0
        for i in range(num_test):
            m = t.random_rotation_matrix()
            m[:3, 3:4] = t.random_vector(3).reshape(3,1)
            m1 = t.inverse_matrix(m)
            m2 = homogeneous_matrix_inverse(m)
            successes += int(np.allclose(m1, m2))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test


    # transform_from_matrix
    if test_enabled[7]:
        print("test transform_from_matrix")
        successes = 0
        for i in range(num_test):
            q0 = t.random_quaternion()
            p0 = t.random_vector(3)
            m0 = homogeneous_matrix(p0, q0)
            t0 = transform_from_matrix(m0)
            p1 = t0[:3]
            q1 = t0[3:]
            successes += int(np.allclose(p0, p1) and (np.allclose(q0, q1) or np.allclose(q0, -q1)))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test


    # transform
    if test_enabled[8]:
        print("test transform")
        successes = 0
        for i in range(num_test):
            q0 = t.random_quaternion()
            p0 = t.random_vector(3)
            world_to_target = p0.tolist() + q0.tolist()

            q1 = t.random_quaternion()
            p1 = t.random_vector(3)
            target_to_offset = p1.tolist() + q1.tolist()
            offset_to_target = transform_inverse(target_to_offset)

            world_to_offset = transform(world_to_target, target_to_offset)
            world_to_target_new = transform(world_to_offset, offset_to_target)

            successes += int(np.allclose(world_to_target[:3], world_to_target_new[:3]) and \
                             (np.allclose(world_to_target[3:], world_to_target_new[3:]) or \
                              np.allclose(world_to_target[3:], -world_to_target_new[3:])))
        print('   -> successes: {}/{}'.format(successes, num_test))
        print("--------------------\n")
        tot_successess += successes
        tot_test += num_test

    
    print("tot successess: {}/{}".format(tot_successess, tot_test))

