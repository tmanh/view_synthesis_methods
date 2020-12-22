import numpy as np
import numba as nb

from numba import jit

import scipy


@jit('Tuple((float32[:,:],float32[:,:]))(float64[:,:,:],uint8[:,:,:])')
def generate_vertices(pos_matrix, image):
    height, width, _ = image.shape

    # build point cloud
    vertices = np.ones((height * width, 3), dtype=np.float32)
    vertex_colors = np.ones((height * width, 4), dtype=np.float32)

    i = 0
    for y in range(height):
        for x in range(width):
            vertices[i, 0] = pos_matrix[0, y, x]
            vertices[i, 1] = pos_matrix[1, y, x]
            vertices[i, 2] = pos_matrix[2, y, x]
            
            """
            vertex_colors[i, 0] = image[y, x, 2]
            vertex_colors[i, 1] = image[y, x, 1]
            vertex_colors[i, 2] = image[y, x, 0]
            vertex_colors[i, 3] = 255
            """

            vertex_colors[i, 0] = image[y, x, 0] / 255
            vertex_colors[i, 1] = image[y, x, 1] / 255
            vertex_colors[i, 2] = image[y, x, 2] / 255

            i += 1

    return vertices, vertex_colors


def generate_faces_from_edges(edges):
    height, width = edges.shape

    faces = np.zeros((0, 3), dtype=np.int32)

    # . - o
    # o - x
    flag_1 = generate_down_right_mask(edges, height, width)
    rows, cols = np.where(flag_1 == True)

    # generate triangles
    down_right = generate_faces_down_right(rows, cols, width)
    faces = np.concatenate([faces, down_right], axis=0)

    # . - o
    # o - x
    flag_2 = generate_up_left_mask(edges, height, width)
    rows, cols = np.where(flag_2 == True)

    # generate triangles
    up_left = generate_faces_up_left(rows, cols, width)
    faces = np.concatenate([faces, up_left], axis=0)

    return faces


@jit('boolean[:,:](uint8[:,:],int32,int32)')
def generate_down_right_mask(quantizied, height, width):
    output = np.zeros((height, width), dtype=nb.boolean)

    for i in range(height - 1):
        for j in range(width - 1):
            if quantizied[i, j] == 0:
                if quantizied[i, j + 1] == 0 and quantizied[i + 1, j] == 0:
                    output[i, j] = True

    return output


@jit('boolean[:,:](uint8[:,:],int32,int32)')
def generate_up_left_mask(quantizied, height, width):
    output = np.zeros((height, width), dtype=nb.boolean)

    for i in range(1, height):
        for j in range(1, width):
            if quantizied[i, j] == 0:
                if quantizied[i, j - 1] == 0 and quantizied[i - 1, j] == 0:
                    output[i, j] = True

    return output


@jit('int32[:,:](int64[:],int64[:],int32)')
def generate_faces_down_right(row_triangles, col_triangles, width):
    faces = np.zeros((len(row_triangles), 3), dtype=np.int32)
    for i in range(len(row_triangles)):
        faces[i, 0] = row_triangles[i] * width + col_triangles[i]
        faces[i, 1] = (row_triangles[i] + 1) * width + col_triangles[i]
        faces[i, 2] = row_triangles[i] * width + col_triangles[i] + 1

        i += 1
    return faces


@jit('int32[:,:](int64[:],int64[:],int32)')
def generate_faces_up_left(row_triangles, col_triangles, width):
    faces = np.zeros((len(row_triangles), 3), dtype=np.int32)
    
    for i in range(len(row_triangles)):
        faces[i, 0] = row_triangles[i] * width + col_triangles[i]
        faces[i, 1] = (row_triangles[i] - 1) * width + col_triangles[i]
        faces[i, 2] = row_triangles[i] * width + col_triangles[i] - 1

    return faces


def create_loc_matrix_from_depth(depth, height, width):
    y = np.arange(0, height).repeat(width, 0).reshape((1, height, width))                    # columns
    x = np.arange(0, width).reshape(1, width).repeat(height, 0).reshape((1, height, width))  # rows
    z = depth.reshape((1, height, width))
    ones = np.ones((1, height, width))

    pos_matrix = np.concatenate([x * z, y * z, z, ones], axis=0)

    return pos_matrix


def create_loc_matrix_from_depth(depth, max_depth=6.0):
    height, width = depth.shape

    depth[depth > max_depth] = max_depth

    y = np.arange(0, height).repeat(width, 0).reshape((1, height, width))                    # columns
    x = np.arange(0, width).reshape(1, width).repeat(height, 0).reshape((1, height, width))  # rows
    z = depth.reshape((1, height, width))
    ones = np.ones((1, height, width))

    pos_matrix = np.concatenate([x * z, y * z, z, ones], axis=0)

    return pos_matrix


def warp(pos_matrix, intrinsic, extrinsic):
    _, height, width = pos_matrix.shape

    pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    pose[:3, :] = np.matmul(intrinsic, extrinsic[:3, :])
    inv_pose = np.linalg.inv(pose)

    pos_matrix = np.matmul(inv_pose, pos_matrix.reshape(4, -1))
    pos_matrix = pos_matrix.reshape((4, height, width))
    
    return pos_matrix


def get_rotation_translation(extrinsic):
    rotation = scipy.spatial.transform.Rotation.from_matrix(extrinsic[:3, :3] @ np.diag([1, -1, -1]))
    rotation = rotation.as_euler('xyz', degrees=True)

    translation = np.array([extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3]])

    return translation, rotation
