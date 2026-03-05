import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """

    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)

    single_point = False
    if pts.ndim == 1:          # handle single point
        pts = pts.reshape(1, 3)
        single_point = True

    # convert to homogeneous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])

    # apply transformation
    transformed = (T @ pts_h.T).T

    # drop last coordinate
    result = transformed[:, :3]

    if single_point:
        return result[0]

    return result