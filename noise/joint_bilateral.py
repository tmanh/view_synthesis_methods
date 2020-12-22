import math
import numpy as np

from numba import jit


def compute_kernel(sigma_space, sigma_color, kernel_size):    
    # gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gauss_space_coeff = -0.5 / (sigma_space ** 2)

    # gaussian function to calcualte the color range kernel
    gauss_color_coeff = -0.5 / (sigma_color ** 2)
    color_weights = np.exp(np.arange(256 * 3) ** 2 * gauss_color_coeff)

    xx = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    x, y = np.meshgrid(xx , xx)
    
    # calculate spatial kernel from the gaussian function.
    # That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2) 
    space_weights = np.exp(-(x ** 2 + y ** 2) * gauss_space_coeff)
    
    return color_weights, space_weights


def joint_bilateral_filter(depth, image, sigma_space=4.0, color_sigma=0.5, kernel_size=7):
    color_weights, space_weights = compute_kernel(sigma_space, color_sigma, kernel_size)

    output = numba_joint_bilateral_filter(depth, image.astype(np.int32), color_weights, space_weights, np.int32(kernel_size))

    return output


def weighted_joint_bilateral_filter(depth, image, sigma_space=4.0, color_sigma=0.5, speckle_sigma=1.4,
                                    kernel_size=7, speckle_window_size=38, speckle_range=100.0, bad_depth_value=0.0):
    color_weights, space_weights = compute_kernel(sigma_space, color_sigma, kernel_size)

    speckle_maps = speckle_filter(depth, speckle_window_size, speckle_range, bad_depth_value)
    speckle_weights = compute_speckle_weight_maps(depth, image.astype(np.int32), color_weights, space_weights, speckle_maps, speckle_sigma, kernel_size)

    output = numba_weighted_joint_bilateral_filter(depth, image.astype(np.int32), color_weights, space_weights, speckle_weights, np.int32(kernel_size))

    return output


@jit('float64[:,:](float64[:,:],int32[:,:,:],float64[:],float64[:,:],int32)')
def numba_joint_bilateral_filter(depth, image, color_weights, space_weights, kernel_size):
    height, width = depth.shape
    radius = np.int32(kernel_size / 2)
    output = np.zeros((height, width), np.float64)  # create a matrix the size of the image

    float_zero = np.float64(0.0)
    int_zero = np.int32(0)
    int_one = np.int32(1)
    int_two = np.int32(2)

    for i in range(height):
        for j in range(width):
            value_sum = float_zero
            weight_sum = float_zero

            b0 = image[i, j, int_zero]
            g0 = image[i, j, int_one]
            r0 = image[i, j, int_two]

            for y in range(-radius, radius + int_one):
                for x in range(-radius, radius + int_one):
                    if int_zero <= i + y < height and int_zero <= j + x < width:
                        b = image[i + y, j + x, int_zero]
                        g = image[i + y, j + x, int_one]
                        r = image[i + y, j + x, int_two]

                        c_w = color_weights[abs(b - b0) + abs(g - g0) + abs(r - r0)]
                        w = space_weights[y + radius, x + radius] * c_w

                        value_sum += w * depth[i + y, j + x]
                        weight_sum += w
            output[i, j] = value_sum / weight_sum
    return output


@jit('int32[:,:](float64[:,:],int32,float64,float64)')
def speckle_filter(depth, speckle_window_size, speckle_range, bad_depth_value):
    height, width = depth.shape

    flag = np.zeros((height, width), np.int32)
    output = np.zeros((height, width), np.int32)

    list_x = np.zeros(height * width, np.int32)
    list_y = np.zeros(height * width, np.int32)
    flag_map = np.zeros(height * width, np.int32)

    current_label = 0
    for y in range(height):
        for x in range(width):
            if depth[y, x] != bad_depth_value:
                if flag[y, x] > 0:
                    output[y, x] = flag_map[flag[y, x]]
                else:
                    current_label += 1
                    flag[y, x] = current_label

                    count = 0
                    n_pixels = 1
                    idx = 0
                    list_x[idx] = x
                    list_y[idx] = y
                    while n_pixels > idx:
                        cx, cy = list_x[idx], list_y[idx]
                        count += 1

                        if cy + 1 < height:
                            not_check = flag[cy + 1, cx] == 0
                            not_bad_value = depth[cy + 1, cx] != 0
                            in_speckle_range = abs(depth[cy + 1, cx] - depth[cy, cx]) <= speckle_range
                            if not_check and not_bad_value and in_speckle_range:
                                flag[cy + 1, cx] = current_label
                                list_x[n_pixels] = cx
                                list_y[n_pixels] = cy + 1
                                n_pixels += 1

                        if cy - 1 >= 0:
                            not_check = flag[cy - 1, cx] == 0
                            not_bad_value = depth[cy - 1, cx] != 0
                            in_speckle_range = abs(depth[cy - 1, cx] - depth[cy, cx]) <= speckle_range
                            if not_check and not_bad_value and in_speckle_range:
                                flag[cy - 1, cx] = current_label
                                list_x[n_pixels] = cx
                                list_y[n_pixels] = cy - 1
                                n_pixels += 1

                        if cx + 1 < width:
                            not_check = flag[cy, cx + 1] == 0
                            not_bad_value = depth[cy, cx + 1] != 0
                            in_speckle_range = abs(depth[cy, cx + 1] - depth[cy, cx]) <= speckle_range
                            if not_check and not_bad_value and in_speckle_range:
                                flag[cy, cx + 1] = current_label
                                list_x[n_pixels] = cx + 1
                                list_y[n_pixels] = cy
                                n_pixels += 1

                        if cx - 1 >= 0:
                            not_check = flag[cy, cx - 1] == 0
                            not_bad_value = depth[cy, cx - 1] != 0
                            in_speckle_range = abs(depth[cy, cx - 1] - depth[cy, cx]) <= speckle_range
                            if not_check and not_bad_value and in_speckle_range:
                                flag[cy, cx - 1] = current_label
                                list_x[n_pixels] = cx - 1
                                list_y[n_pixels] = cy
                                n_pixels += 1

                        idx += 1
                    
                    if count <= speckle_window_size:
                        flag_map[current_label] = 0
                        output[y, x] = 0
                    else:
                        flag_map[current_label] = 1
                        output[y, x] = 1

    return output


@jit('float64[:,:](float64[:,:],int32[:,:,:],float64[:],float64[:,:],int32[:,:],float64,int32)')
def compute_speckle_weight_maps(depth, image, color_weights, space_weights, speckle_maps, speckle_sigma, kernel_size):
    height, width = depth.shape
    radius = np.int32(kernel_size / 2)

    output = np.zeros((height, width), np.float64)  # create a matrix the size of the image

    for i in range(height):
        for j in range(width):
            value_sum = 0.0

            b0 = image[i, j, 0]
            g0 = image[i, j, 1]
            r0 = image[i, j, 2]

            for y in range(-radius, radius + 1):
                for x in range(-radius, radius + 1):
                    if 0 <= i + y < height and 0 <= j + x < width:
                        b = image[i + y, j + x, 0]
                        g = image[i + y, j + x, 1]
                        r = image[i + y, j + x, 2]

                        c_w = color_weights[abs(b - b0) + abs(g - g0) + abs(r - r0)]
                        w = space_weights[y + radius, x + radius] * c_w * speckle_maps[i, j]
                        
                        value_sum += w * np.exp(-0.5 * (depth[i + y, j + x] - depth[i + y, j + x]) ** 2 / speckle_sigma)

            output[i, j] = value_sum

    return output


@jit('float64[:,:](float64[:,:],int32[:,:,:],float64[:],float64[:,:],float64[:,:],int32)')
def numba_weighted_joint_bilateral_filter(depth, image, color_weights, space_weights, speckle_weights, kernel_size):
    height, width = depth.shape
    radius = np.int32(kernel_size / 2)
    output = np.zeros((height, width), np.float64)  # create a matrix the size of the image

    float_zero = np.float64(0.0)

    for i in range(height):
        for j in range(width):
            value_sum = float_zero
            weight_sum = float_zero

            b0 = image[i, j, 0]
            g0 = image[i, j, 1]
            r0 = image[i, j, 2]

            for y in range(-radius, radius + 1):
                for x in range(-radius, radius + 1):
                    if 0 <= i + y < height and 0 <= j + x < width:
                        b = image[i + y, j + x, 0]
                        g = image[i + y, j + x, 1]
                        r = image[i + y, j + x, 2]

                        c_w = color_weights[abs(b - b0) + abs(g - g0) + abs(r - r0)]
                        w = space_weights[y + radius, x + radius] * speckle_weights[i + y, j + x] * c_w
                        
                        w = w if w != 0 else 1e-7

                        value_sum += w * depth[i + y, j + x]
                        weight_sum += w
            output[i, j] = value_sum / weight_sum
    return output
