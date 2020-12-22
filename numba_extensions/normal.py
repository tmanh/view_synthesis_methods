import math
import numpy as np

from numba import jit


def compute_normal(depth):
    zy, zx = np.gradient(depth)  
    
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))

    n = np.linalg.norm(normal, axis=2) + 1e-7
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    # normal += 1
    # normal /=_ 2
    # normal *= 255

    return normal  # [:, :, ::-1]


@jit('float64(float64[:],float64[:])')
def dot_product(v1, v2):
    output = 0
    for i in range(len(v1)):
        output += v1[i] * v2[i]
    return output


@jit('float64(float64[:])')
def length(v):
    return math.sqrt(dot_product(v, v))


@jit('float64(float64[:],float64[:])')
def angle(v1, v2):
    return math.acos(dot_product(v1, v2) / (length(v1) * length(v2) + 1e-7))


@jit('float64[:,:](float64[:,:,:],int32,int32)')
def compute_angles_from(normal, height, width):
    output = np.zeros((height, width), dtype=np.float64)

    oz = np.zeros((3,), dtype=np.float64)
    oz[2] = 1.0

    for i in range(height):
        for j in range(width):
            output[i, j] = angle(normal[i, j], oz)

    return output


def compute_angles(depth):
    height, width = depth.shape
    normal = compute_normal(depth)
    output = compute_angles_from(normal, height, width)
    return output
