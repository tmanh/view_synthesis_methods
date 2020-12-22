import cv2
import math
import numpy as np

from numba import jit



def compute_edges(depth, upper_bound_depth=10000):
    max_depth = depth.max()
    if max_depth >= upper_bound_depth:
        max_depth = upper_bound_depth
    
    norm_bilateral = (depth - depth.min()) / max(max_depth - depth.min(), 1e-7)

    horizon_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 0)) > 0.035).astype(np.uint8) * 255
    vertical_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 1)) > 0.035).astype(np.uint8) * 255
    edges = np.maximum(horizon_check, vertical_check)

    return edges


@jit('Tuple((float32,float32))(float32,float32,float32,float32,boolean)')
def compute_sigma(phi, z, ratio=np.float32(1/150), scale = 0.1, assume_in_range=True):
    # ratio = size of pixel / focal length
    # Ai2thor: FOV (default 90) = 2*arctan(pixelNumber/(2*focalLength))
    # pixelNumber: image Resolution (width or hight)
    # => focalLength = pixelNumber / tan(90 / 2) / 2
    # => focalLength = 300 / 1 / 2 = 150
    # => size of pixel = 1 => ratio=1/150
    sigma_lateral = (0.8 + 0.035 * phi / (math.pi / 2 - phi)) * z * ratio * scale
    sigma_axial = 0.0012 + 0.0019 * (z - 0.4) ** 2  # z
    if not assume_in_range:
        sigma_axial += 0.0001 * phi ** 2 / math.sqrt(z) / (math.pi / 2 - phi) ** 2

    # sigma_lateral *= 1000
    # sigma_axial *= 1000

    return sigma_axial, sigma_lateral


@jit('float64[:,:](float64[:,:],uint8[:,:],float64[:,:])')
def add_noise(depth, edges, angles):
    height, width = depth.shape

    # assume the angles vary from 10-60
    output = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            sigma_axial, sigma_lateral = compute_sigma(angles[i, j], depth[i, j], np.float32(1/300), 0.1, True)
            axial_noise = np.random.normal(loc=0, scale=sigma_axial)
                
            output[i, j] = depth[i, j] + axial_noise

            # """
            if edges[i, j] == 0:
                lateral_noise = np.random.normal(loc=0, scale=sigma_lateral)
                output[i, j] = output[i, j] + lateral_noise
            # """

    return output


def generate_edge_map(edges, iterations=5):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=iterations)

    return dilation
