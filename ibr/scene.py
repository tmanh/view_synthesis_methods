import os
import cv2

import numpy as np

from noise.model import compute_edges, add_noise
from noise.joint_bilateral import joint_bilateral_filter
# from noise.sparse_bilateral import sparse_bilateral_filtering

from ibr.mesh import CanvasView

from numba_extensions.normal import compute_angles
from numba_extensions.scene import generate_vertices, generate_faces_from_edges
from numba_extensions.scene import create_loc_matrix_from_depth, warp, get_rotation_translation


class Scene:
    def __init__(self):
        super().__init__()

        self.images = []
        self.depths = []
        self.extrinsics = []

        self.mesh = []

        self.intrinsic = np.array([[300, 0, 300], [0, 300, 300], [0, 0, 1]])

    def set_data(self, images, depths, extrinsics, intrinsic):
        for i in range(len(images)):
            self.images.append(images[i])
            self.depths.append(depths[i])
            self.extrinsics.append(extrinsics[i])

        self.intrinsic = intrinsic

    def read_data(self, data_dir, scene_name, n_views):
        for i in range(n_views):
            img = cv2.imread(os.path.join(data_dir, scene_name + '_i_{}.png'.format(i)))
            depth = np.load(os.path.join(data_dir, scene_name + '_d_{}.npy'.format(i)))
            extrinsic = np.load(os.path.join(data_dir, scene_name + '_p_{}.npy'.format(i)))
            extrinsic[:3, 3] = extrinsic[:3, 3] / 1000
            
            self.extrinsics.append(extrinsic)
            self.images.append(img)
            self.depths.append(depth.astype(np.float64))

    def visualize(self):
        for i in range(len(self.images)):
            cv2.imshow('image_{}'.format(i), self.images[i])

            norm_depth = (self.depths[i] - self.depths[i].min()) / (self.depths[i].max() - self.depths[i].min()) * 255
            norm_depth = norm_depth.astype(np.uint8)
            cv2.imshow('depth_{}'.format(i), norm_depth)

            if i < len(self.mesh):
                cv2.imshow('meshed_{}'.format(i), self.mesh[i])

        cv2.waitKey()
    
    def add_noise(self):
        for i in range(len(self.depths)):
            edges = compute_edges(self.depths[i])
            angles = compute_angles(self.depths[i])

            self.depths[i] = add_noise(self.depths[i], edges, angles)
            print(self.depths[i].min(), self.depths[i].max())

    def denoise(self):
        for i in range(len(self.images)):
            # _, bilateral = sparse_bilateral_filtering(self.depths[i], self.images[i])
            # self.depths[i] = bilateral[-1]
            self.depths[i] = joint_bilateral_filter(self.depths[i], self.images[i])

    def create_mesh(self, extrinsic):
        for i in range(len(self.images)):
            edges = compute_edges(self.depths[i])

            pos_matrix = create_loc_matrix_from_depth(self.depths[i])
            pos_matrix = warp(pos_matrix, self.intrinsic, self.extrinsics[i])
            
            vertices, vertex_colors = generate_vertices(pos_matrix, self.images[i])
            faces = generate_faces_from_edges(edges)

            translation, rotation = get_rotation_translation(self.extrinsics[i])

            scene = CanvasView(90, vertices, faces, vertex_colors, translation, rotation)

            translation, rotation = get_rotation_translation(extrinsic)
            scene.transform(translation, rotation)

            render = scene.render()[:, ::-1, :]
            self.mesh.append(render)
