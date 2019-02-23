import numpy as np


def iceil(x):
    return np.ceil(x).astype(np.int32)


def cos_angle(v0, v1):
    # return the cosine of the angle.
    norm = np.linalg.norm
    return np.dot(v0, v1) / (norm(v0) * norm(v1))
