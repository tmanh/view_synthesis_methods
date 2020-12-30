import os
import cv2

import numpy as np

from ..noise.model import compute_edges, add_noise
from ..noise.joint_bilateral import joint_bilateral_filter

from ..ibr.mesh import ViewRenderer

from ..numba_extensions.normal import compute_angles
from ..numba_extensions.scene import generate_vertices, generate_faces_from_edges
from ..numba_extensions.scene import create_loc_matrix_from_depth, warp, get_rotation_translation


class Scene:
    def __init__(self, resolution, fov):
        super().__init__()

        self.images = []
        self.depths = []
        self.extrinsics = []

        self.warp_colors = []
        self.warp_depths = []

        self.resolution = resolution
        self.fov = fov

        self.intrinsic = np.array([[300, 0, 300], [0, 300, 300], [0, 0, 1]])

    def set_data(self, images, depths, extrinsics, intrinsic):
        self.extrinsics.clear()
        self.images.clear()
        self.depths.clear()

        for i in range(len(images)):
            self.images.append(images[i].astype(np.uint8))
            self.depths.append(depths[i].astype(np.float64))
            self.extrinsics.append(extrinsics[i])

        self.intrinsic = intrinsic

    def read_data(self, data_dir, scene_name, n_views):
        self.extrinsics.clear()
        self.images.clear()
        self.depths.clear()

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

            if i < len(self.warp_colors):
                cv2.imshow('warped_color_{}'.format(i), self.warp_colors[i])

            if i < len(self.warp_depths):
                norm_depth = (self.warp_depths[i] - self.warp_depths[i].min()) / (self.warp_depths[i].max() - self.warp_depths[i].min()) * 255
                norm_depth = norm_depth.astype(np.uint8)
                cv2.imshow('warped_depth_{}'.format(i), norm_depth)

        cv2.waitKey()
    
    def add_noise(self):
        for i in range(len(self.depths)):
            edges = compute_edges(self.depths[i])
            angles = compute_angles(self.depths[i])

            self.depths[i] = add_noise(self.depths[i], edges, angles)

    def denoise(self):
        for i in range(len(self.images)):
            # _, bilateral = sparse_bilateral_filtering(self.depths[i], self.images[i])
            # self.depths[i] = bilateral[-1]
            self.depths[i] = joint_bilateral_filter(self.depths[i], self.images[i])

    def create_mesh(self, extrinsic):
        for i in range(len(self.images)):
            min_depth, max_depth = self.depths[i].min(), self.depths[i].max()
            near = 10 ** np.floor(np.log10(min_depth))
            far = 10 ** np.ceil(np.log10(max_depth))

            edges = compute_edges(self.depths[i])

            pos_matrix = create_loc_matrix_from_depth(self.depths[i])
            pos_matrix = warp(pos_matrix, self.intrinsic, self.extrinsics[i])
            
            vertices, vertex_colors = generate_vertices(pos_matrix, self.images[i])
            faces = generate_faces_from_edges(edges)

            translation, rotation = get_rotation_translation(extrinsic)

            renderer = ViewRenderer(self.fov, self.resolution, vertices, faces, vertex_colors, translation, rotation, near, far)
            warp_color, warp_depth = renderer.render()

            self.warp_colors.append(warp_color[:, ::-1, :][:, :, :3].copy())
            self.warp_depths.append(warp_depth[:, ::-1, :].copy())
