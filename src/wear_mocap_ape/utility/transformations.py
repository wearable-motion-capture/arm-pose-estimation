import math

import numpy as np


def moving_average(x, w) -> np.array:
    return np.convolve(x, np.ones(w), 'same') / w


def derive_series(y, x, axis=0) -> np.array:
    """
    :param y: series to derive
    :param x: time stamps, must be a 1D array
    :param axis:
    :return: derivative of len(y) -1
    """
    return np.diff(y, axis=axis) / np.diff(x)


def integrate_series(start_y, y, x) -> np.array:
    """
    :param start_y: first value has to be known
    :param y: series to integrate
    :param x: time stamps
    :return: integrated series on length len(y) + 1
    """
    c_y = np.diff(x) * y
    c_y = np.insert(c_y, 0, start_y)  # prepend start value
    return np.cumsum(c_y)


def average_quaternions(quats: np.array):
    """
    This averaging only works if the quaternions are reasonably close together.
    This function assumes that the quaternions
    come in a np array [x, 4] where x is the number of quaternions in the array
    :param quats: [x, 4] where dim 1 is expected in the order w,x,y,z
    :return: averaged quaternion
    """
    w = 1 / len(quats)
    q0 = quats[0, :]
    qavg = q0 * w
    for qi in quats[1:, :]:
        if np.dot(qi, q0) < 0.0:
            # two quaternions can represent the same orientation
            # "flip" them back close to the first quaternion if needed
            qavg += qi * -w
        else:
            # otherwise, just average
            qavg += qi * w
    return qavg / np.linalg.norm(qavg)  # normalize again


def scale_vector_columns(vecs: np.array, magnitudes) -> np.array:
    """
    Designed to change the magnitude of many vectors that come in three columns [[x,y,z],[x,y,z],...]. According to an
    equally long list of magnitudes [m1, m2, ... ].  Just in case data comes in various shapes during a procedure,
    this function also deals with a single vec [x,y,z].
    :param vecs: [[x,y,z],[x,y,z], ... ]
    :param magnitudes: single vec of target magnitudes [ m1, m2, ... ]
    :return: normalized vecs
    """
    # normalize
    n_vecs = normalize_vector_columns(vecs)
    # multiply with magnitude
    return n_vecs * np.array(magnitudes)[..., np.newaxis]


def normalize_vector_columns(vecs: np.array) -> np.array:
    """
    Designed to ease the normalization of many vectors that come in three columns [[x,y,z],[x,y,z],...].
    Just in case data comes in various shapes during a procedure, this function also deals with a single vec [x,y,z].
    :param vecs:
    :return: normalized vecs
    """
    if len(vecs.shape) > 1:
        rs = np.sqrt(np.sum(np.square(vecs), axis=1))[..., np.newaxis]
        return np.divide(vecs, rs)
    else:
        return np.divide(vecs, np.linalg.norm(vecs))


def quat_rotate_vector(rot: np.array, vec: np.array) -> np.array:
    """
    To get the rotated vector do this:

    H is the hamilton product
    P  = [0, p1, p2, p3] is a point vector
    R  = [w,  x,  y,  z] is a rotation
    R' = [w, -x, -y, -z]

    P' = RPR' is the rotated vector
    P' = H(H(R, P), R') do this
    :param vec: Vec4 as [0,x,y,z]
    :param rot: Quat as [w,x,y,z]
    :return: rotated vector as [x,y,z]
    """
    if len(rot.shape) > 1:
        if rot.shape[1] != 4:
            raise UserWarning("rot has to have length 4 (w,x,y,z). Rot is {}".format(rot))
    elif len(rot) != 4:
        raise UserWarning("rot has to have length 4 (w,x,y,z). Rot is {}".format(rot))

    # the conjugate of the quaternion
    r_s = rot * np.array([1, -1, -1, -1])

    if len(vec.shape) > 1:
        # if vec is a column of vectors
        if vec.shape[1] != 3:
            raise UserWarning("vec has to have length 3 (x,y,z). Vec is {}".format(vec))
        # prepend a 0
        vec = np.insert(vec, 0, 0, axis=1)
        return hamilton_product(hamilton_product(rot, vec), r_s)[:, 1:]  # remove first 0
    elif len(vec) != 3:
        # if vec has wrong length
        raise UserWarning("vec has to have length 3 (x,y,z). Vec is {}".format(vec))
    else:
        # if vec is a single vector
        vec = np.insert(vec, 0, 0)  # prepend a 0

        rotated_vec = hamilton_product(hamilton_product(rot, vec), r_s)
        # the rotated vector might be an entire data column if we applied a column of rotations
        if len(rotated_vec.shape) > 1:
            return rotated_vec[:, 1:]  # remove first 0s
        else:
            return rotated_vec[1:]  # remove first 0


def hamilton_product(a: np.array, b: np.array) -> np.array:
    """
    Hamilton product for two quaternions or a Vec4 and a Quaternion.
    :param a: quaternion in order [w,x,y,z] or Vec4 as [0,x,y,z]
    :param b: quaternion in order [w,x,y,z]
    """
    # check shape to deal with a whole column of rotations
    if len(a.shape) > 1:
        a = [a[:, 0], a[:, 1], a[:, 2], a[:, 3]]
    if len(b.shape) > 1:
        b = [b[:, 0], b[:, 1], b[:, 2], b[:, 3]]
    h_p = np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    ])
    if len(h_p.shape) > 1:
        return h_p.transpose()
    else:
        return h_p


def euler_to_quat(e: np.array) -> np.array:
    # make sure the code can deal with a whole data column of quaternions
    if len(e.shape) > 1:
        e = [e[:, 0], e[:, 1], e[:, 2]]

    cr = np.cos(e[0] * 0.5)
    sr = np.sin(e[0] * 0.5)
    cp = np.cos(e[1] * 0.5)
    sp = np.sin(e[1] * 0.5)
    cy = np.cos(e[2] * 0.5)
    sy = np.sin(e[2] * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])

    if len(q.shape) > 1:
        return q.transpose()
    else:
        return q


def hips_sin_cos_to_quat(hips_sin, hips_cos):
    y_rot = np.arctan2(hips_sin, hips_cos)
    return euler_to_quat(np.c_[np.zeros(y_rot.shape), y_rot, np.zeros(y_rot.shape)])


def calib_watch_left_to_north_quat(sw_quat_fwd: np.array) -> float:
    """
    Estimates rotation around global y-axis (Up) from watch orientation.
    This corresponds to the azimuth in polar coordinates. If the angle from the z-axis (forward)
    in between +pi and -pi.
    :param sw_quat_fwd: quat as [w,x,y,z] (in sw coord system)
    :return: north quat which rotates around the y-axis until global Z and North are aligned
    """
    # smartwatch rotation to global coordinates, which are [-w,x,z,y]
    r = android_quat_to_global_no_north(sw_quat_fwd)
    y_rot = reduce_global_quat_to_y_rot(r)
    if len(r.shape) > 1:
        q_north = euler_to_quat(np.c_[np.zeros(y_rot.shape), -y_rot, np.zeros(y_rot.shape)])
    else:
        q_north = euler_to_quat(np.array([0, -y_rot, 0]))
    return hamilton_product(np.array([0.7071068, 0, -0.7071068, 0]), q_north)  # rotation to match left hand calibration


def reduce_global_quat_to_y_rot(q: np.array):
    p = np.array([0, 0, 1])  # forward vector with [x,y,z]
    pp = quat_rotate_vector(q, p)
    # get angle with atan2
    if len(q.shape) > 1:
        return np.arctan2(pp[:, 0], pp[:, 2])
    else:
        return np.arctan2(pp[0], pp[2])


def calib_watch_right_to_north_quat(sw_quat_fwd: np.array) -> float:
    """
    Estimates rotation around global y-axis (Up) from watch orientation.
    This corresponds to the azimuth in polar coordinates. If the angle from the z-axis (forward)
    in between +pi and -pi.
    :param sw_quat_fwd: quat as [w,x,y,z] (in sw coord system)
    :return: north quat which rotates around the y-axis until global Z and North are aligned
    """
    # smartwatch rotation to global coordinates, which are [-w,x,z,y]
    r = android_quat_to_global_no_north(sw_quat_fwd)
    y_rot = reduce_global_quat_to_y_rot(r)
    q_north = euler_to_quat(np.array([0, -y_rot, 0]))
    return hamilton_product(np.array([0.7071068, 0, 0.7071068, 0]), q_north)  # rotation to match right hand calibration


def android_quat_to_global_no_north(q: np.array) -> np.array:
    if len(q.shape) > 1:
        return q[:, [0, 1, 3, 2]] * np.array([-1, 1, 1, 1])
    else:
        return np.array([-q[0], q[1], q[3], q[2]])


def android_quat_to_global(q: np.array, north_quat: np.array) -> np.array:
    """
    sw coord system is X East, Y North, Z Up (right-hand).
    This function changes it to X Right, Z Forward, Y Up (left-hand) in our global coord.
    :param q: sw quat as [w,x,y,z]
    :param north_quat: quat to align magnetic North with global Z-axis
    :return: sw global quat as [w,x,y,z]
    """
    qn = android_quat_to_global_no_north(q)
    return hamilton_product(north_quat, qn)


def quat_invert(q: np.array):
    """
    estimates the inverse rotation.
    :param q: input quaternion
    :return: inverse quaternion
    """
    q_s = q * np.array([1, -1, -1, -1])  # the conjugate of the quaternion
    if len(q.shape) > 1:
        return q_s / np.sum(np.square(q), axis=1, keepdims=True)
    else:
        return q_s / np.sum(np.square(q))


def quat_to_euler(q: np.array):
    """
    from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    :param q: quaternion as array [w,x,y,z]
    :return: euler angles in array [x,y,z]
    """

    # deal with entire columns of rotations
    if len(q.shape) > 1:
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    a_x = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    # use 90 degrees if out of range
    if len(q.shape) > 1:
        a_y = np.zeros(len(sinp))
        for i in range(len(sinp)):
            if np.abs(sinp[i]) >= 1:
                a_y[i] = np.copysign(math.pi / 2, sinp[i])
            else:
                a_y[i] = np.arcsin(sinp[i])
    else:
        if np.abs(sinp) >= 1:
            a_y = np.copysign(math.pi / 2, sinp)
        else:
            a_y = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    a_z = np.arctan2(siny_cosp, cosy_cosp)

    if len(q.shape) > 1:
        return np.column_stack([a_x, a_y, a_z])
    else:
        return np.array([a_x, a_y, a_z])


def mocap_pos_to_global(p: np.array):
    """
    mocap coord system to unity coord system
    :param p: position as [x,y,z]
    :return:  position as [x,y,z]
    """
    # swap x and z axis
    if len(p.shape) > 1:
        p[:, [0, 2]] = p[:, [2, 0]]
    else:
        p[[0, 2]] = p[[2, 0]]
    # invert x and z axis
    return p * np.array([-1, 1, -1])


def mocap_quat_to_global(q: np.array):
    """
    Optitrack quaternion to rotation in world coords
    :param q: quaternion as [x,y,z,w]
    :return: quaternion as [w,x,y,z]
    """
    # bring w to front
    if len(q.shape) > 1:
        mc_q = np.array([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
    else:
        mc_q = np.array([q[3], q[0], q[1], q[2]])
    # rotate by -90 around y-axis to align z-axis of both coord systems ...
    wordl_quat = np.array([-0.7071068, 0, 0.7071068, 0])
    # then flip x-axis and reverse angle to change coord system orientation
    n_q = mc_q * np.array([-1, -1, 1, 1])
    return hamilton_product(wordl_quat, n_q)


def quat_a_to_b(vec_a: np.array, vec_b: np.array):
    """
    The rotation between two vectors as a rotation
    :param vec_a: [x,y,z]
    :param vec_b: [x,y,z]
    :return: q[w,x,y,z]
    """

    if len(vec_a.shape) > 1 and len(vec_b.shape) > 1:
        # normalize vectors since we only care about rotation
        ms_a = np.sqrt(np.sum(np.square(vec_a), axis=1))[..., np.newaxis]
        ms_b = np.sqrt(np.sum(np.square(vec_b), axis=1))[..., np.newaxis]

        n_a = np.divide(vec_a, ms_a)
        n_b = np.divide(vec_b, ms_b)

        w = np.zeros((len(n_a), 1))
        for n in range(len(n_a)):
            w[n, 0] = np.dot(n_a[n, :], n_b[n, :]) + 1

        # c = np.tensordot(n_a, n_b, axes=[[1], [1]])[0, :, np.newaxis] + 1.

        cross_idx = np.sum(w, axis=1) != 0
        xyz = np.cross(n_a[cross_idx], n_b[cross_idx], axis=1)

        ndeg = np.array([[0, 2., 0]]).repeat(len(n_a), axis=0)
        ndeg[cross_idx] = xyz

        q = np.hstack([w, ndeg]) / 2.
        return q / np.linalg.norm(q, axis=1)[:, np.newaxis]  # normalize again
    else:
        n_a = vec_a / np.linalg.norm(vec_a)
        n_b = vec_b / np.linalg.norm(vec_b)
        w = np.dot(n_a, n_b)

        if w > 0.9999 or w < -0.9999:
            return np.array([0, 0, 1, 0])  # 180-degree rotation around y

        xyz = np.cross(n_a, n_b)

        q = np.insert(xyz, 0, 1. + w) / 2.
        return q / np.linalg.norm(q)  # normalize again


def quaternion_to_exponential_map(q: np.array):
    """
    from https://turtle.codes/projects/exponentialmaps.html
    :param q: array as w,x,y,z
    :return:
    """
    if len(q.shape) > 1:
        q_w = q[:, 0]  # angle
        q_img = q[:, 1:]  # imaginary values x,y,z
    else:
        q_w = q[0]
        q_img = q[1:]

    clip_w = np.clip(q_w, -1, 1)
    theta = np.array(2. * np.arccos(clip_w))

    # transpose to allow multiplications and divisions
    axis = np.transpose(q_img)
    axis = np.divide(axis, np.sin(.5 * theta))
    axis = np.divide(axis, np.linalg.norm(axis, axis=0))  # normalise axis
    axis = np.multiply(axis, theta)
    axis = np.transpose(axis)

    res = np.zeros(axis.shape)
    # leave [0,0,0,0] where we're too close to the zero parameter vector or angle
    idx = np.logical_and(theta > np.finfo(float).eps, np.any(np.abs(q_img) > np.finfo(float).eps))
    res[idx] = axis[idx]

    return res


def vec_to_polar_angles(vec):
    """
    :param vec: in [x,y,z]
    :return: angles [elevation, azimuth, radius]
    """
    if len(vec.shape) > 1:
        rs = np.sqrt(np.sum(np.square(vec), axis=1))[..., np.newaxis]
        norm_vec = np.divide(vec, rs)
        # elevation for elevation angle defined from y-axis downwards
        elevation = np.arctan2(np.sqrt(norm_vec[:, 0] ** 2 + norm_vec[:, 2] ** 2),
                               norm_vec[:, 1])
        # azimuth angle from z-axis (forward)
        azimuth = np.arctan2(norm_vec[:, 0], norm_vec[:, 2])
        return np.column_stack([elevation, azimuth, rs])
    else:
        r = np.sqrt(np.sum(np.square(vec)))
        norm_vec = np.divide(vec, r)
        elevation = np.arctan2(np.sqrt(norm_vec[0] ** 2 + norm_vec[2] ** 2),
                               norm_vec[1])
        azimuth = np.arctan2(norm_vec[0], norm_vec[2])
        return np.column_stack([elevation, azimuth, r])


def polar_angles_to_vec(polar_angles):
    """
    :param polar_angles: [elevation, azimuth]
    :return: vec [x,y,z] assumes magnitude of 1
    """
    if len(polar_angles.shape) > 1:
        return np.column_stack([
            abs(np.sin(polar_angles[:, 0])) * np.sin(polar_angles[:, 1]),
            np.cos(polar_angles[:, 0]),
            abs(np.sin(polar_angles[:, 0])) * np.cos(polar_angles[:, 1]),
        ])
    else:
        return np.column_stack([
            abs(np.sin(polar_angles[0])) * np.sin(polar_angles[1]),
            np.cos(polar_angles[0]),
            abs(np.sin(polar_angles[0])) * np.cos(polar_angles[1]),
        ])


def exponential_map_to_quaternion(em: np.array):
    """
    from https://turtle.codes/projects/exponentialmaps.html
    :param em: array as x,y,z
    :return:
    """
    theta = np.linalg.norm(em)  # angle is the length of the vector

    if theta < np.power(np.finfo(float).eps, .25):
        two_sinc_half_theta = .5 - math.pow(theta, 2.) / 48.
    else:
        two_sinc_half_theta = np.sin(.5 * theta) / theta

    q_img = two_sinc_half_theta * em  # imaginary values x,y,z
    w = np.cos(.5 * theta)
    # prepend w to x,y,z to get full quaternion
    return np.insert(q_img, 0, w)


def six_drr_1x6_to_quat(six_drr: np.array):
    rmat = six_drr_1x6_to_rot_mat_1x9(six_drr)
    return rot_mat_1x9_to_quat(rmat)


def quat_to_6drr_1x6(quat: np.array):
    rmat = quat_to_rot_mat_1x9(quat)
    return rot_mat_1x9_to_six_drr_1x6(rmat)


def quat_to_rot_mat_1x9(quat: np.array):
    """
    transformation of quaternion to rotation matrix. Can deal with lists of quaternions
    :param quat: [w,x,y,z]
    :return: flattened rot mat representations
    """

    # from transforms3d
    def trans(q):
        w, x, y, z = q
        nq = w * w + x * x + y * y + z * z
        if nq < np.finfo(np.float64).eps:
            return np.eye(3)
        s = 2.0 / nq
        _x = x * s
        _y = y * s
        _z = z * s
        w_x = w * _x
        w_y = w * _y
        w_z = w * _z
        x_x = x * _x
        x_y = x * _y
        x_z = x * _z
        y_y = y * _y
        y_z = y * _z
        z_z = z * _z
        return np.array(
            [[1.0 - (y_y + z_z), x_y - w_z, x_z + w_y],
             [x_y + w_z, 1.0 - (x_x + z_z), y_z - w_x],
             [x_z - w_y, y_z + w_x, 1.0 - (x_x + y_y)]])

    if len(quat.shape) > 1:
        ms = []
        for qi in quat:
            ms.append(trans(qi).flatten())
        return np.array(ms)
    else:
        return trans(quat).flatten()


def rot_mat_to_quat(rot_mat: np.array):
    """
    FROM TRANSFORMS3D. See their docu
    Calculate quaternion corresponding to given rotation matrix
    """
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    qxx, qyx, qzx, qxy, qyy, qzy, qxz, qyz, qzz = rot_mat.flat
    # Fill only lower half of symmetric matrix
    k = np.array([
        [qxx - qyy - qzz, 0, 0, 0],
        [qyx + qxy, qyy - qxx - qzz, 0, 0],
        [qzx + qxz, qzy + qyz, qzz - qxx - qyy, 0],
        [qyz - qzy, qzx - qxz, qxy - qyx, qxx + qyy + qzz]]
    ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(k)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q


def rot_mat_3x3_to_six_drr_3x2(rot_mat: np.array):
    """
    See https://arxiv.org/pdf/1812.07035.pdf
    :param rot_mat: 3x3 rotation matrix
    :return: six_dof: [[m1],[m2]] where m1 and m2 are rotation matrix columns
    """
    return rot_mat[:, :2]


def six_drr_3x2_to_rot_mat_3x3(six_dof: np.array):
    """
    See https://arxiv.org/pdf/1812.07035.pdf
    :param six_dof: [[m1],[m2]] where m1 and m2 are rotation matrix columns
    :return: rot_mat: 3x3 rotation matrix
    """

    def n(a: np.array):
        return np.divide(a, np.linalg.norm(a))

    a1, a2 = six_dof[:, 0], six_dof[:, 1]
    b1 = n(a1)
    b2 = n(a2 - (np.dot(b1, a2) * b1))
    b3 = np.cross(b1, b2)

    return np.transpose(np.array([b1, b2, b3]))


def rot_mat_1x9_to_quat(rot_mat: np.array):
    # if its multiple flat rotation matrices in a column
    if len(rot_mat.shape) > 1:
        quats = np.zeros((rot_mat.shape[0], 4))
        # TODO: see if you can speed up this handling. Unfortunately, rot_mat to quaternion isn't a linear transformation
        for i, r_r in enumerate(rot_mat):
            quats[i, :] = rot_mat_to_quat(r_r.reshape((3, 3)))
        return quats
    else:
        return rot_mat_to_quat(rot_mat.reshape((3, 3)))


def rot_mat_1x9_to_six_drr_1x6(rot_mat: np.array):
    """
    See https://arxiv.org/pdf/1812.07035.pdf
    Uses flattened six dof and returns a flattened matrix. This function can deal with columns of multiple rot mats
    :param rot_mat:  [r11,12,r13, r21,r22,r23, r31,r32,r33] rotation matrix
    :return: ix_dof: [m11,m12,m21,m22,m31,m32] where m1 and m2 are rotation matrix columns
    """
    if len(rot_mat.shape) > 1:
        return np.array([rot_mat[:, 0], rot_mat[:, 1],
                         rot_mat[:, 3], rot_mat[:, 4],
                         rot_mat[:, 6], rot_mat[:, 7]]).T
    else:
        return np.array([rot_mat[0], rot_mat[1], rot_mat[3], rot_mat[4], rot_mat[6], rot_mat[7]])


def six_drr_1x6_to_rot_mat_1x9(six_dof: np.array):
    """
    See https://arxiv.org/pdf/1812.07035.pdf
    Uses flattened six dof and returns a flattened matrix. This function can deal with columns of multiple 6dofs
    :param six_dof: [m11,m12,m21,m22,m31,m32] where m1 and m2 are rotation matrix columns
    :return: rot_mat: [r11,12,r13, r21,r22,r23, r31,r32,r33]
    """

    def n(a: np.array):
        if len(a.shape) > 1:
            return np.divide(a.T, np.linalg.norm(a, axis=1)).T
        else:
            return np.divide(a, np.linalg.norm(a))

    if len(six_dof.shape) > 1:
        a1 = np.array([six_dof[:, 0], six_dof[:, 2], six_dof[:, 4]]).T
        a2 = np.array([six_dof[:, 1], six_dof[:, 3], six_dof[:, 5]]).T

        b1 = n(a1)
        b2 = n(a2 - (np.sum(np.multiply(b1, a2), axis=1) * b1.T).T)
        b3 = np.cross(b1, b2, axis=1)

        return np.array([b1[:, 0], b2[:, 0], b3[:, 0],
                         b1[:, 1], b2[:, 1], b3[:, 1],
                         b1[:, 2], b2[:, 2], b3[:, 2]]).T
    else:
        a1 = np.array([six_dof[0], six_dof[2], six_dof[4]])
        a2 = np.array([six_dof[1], six_dof[3], six_dof[5]])

        b1 = n(a1)
        b2 = n(a2 - (np.dot(b1, a2) * b1))
        b3 = np.cross(b1, b2)

        return np.array([b1[0], b2[0], b3[0],
                         b1[1], b2[1], b3[1],
                         b1[2], b2[2], b3[2]])
