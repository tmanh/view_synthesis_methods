import numpy as np

from vispy import scene, io
from vispy.scene import visuals
from vispy.visuals.filters import Alpha


class Mesh:
    def __init__(self, vertices, faces, vertex_colors):
        super().__init__()

        self.vertices = vertices
        self.faces = faces
        self.vertex_colors = vertex_colors

    def apply_transform(self, matrix):
        # check to see if we've been passed an identity matrix
        identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
        if identity < 1e-8:
            return

        self.vertices = np.dot(matrix, self.vertices.T).T


class CanvasView():
    def __init__(self, fov, verts, faces, colors, translation, rotation):
        self.canvas = scene.SceneCanvas(bgcolor='black', size=(600, 600))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'perspective'
        self.view.camera.fov = fov
        self.mesh = visuals.Mesh(shading=None)
        self.mesh.attach(Alpha(1.0))
        self.view.add(self.mesh)
        self.tr = self.view.camera.transform
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])
        self.translate(translation)

        self.rotate(axis=[1, 0, 0], angle=rotation[0])
        self.rotate(axis=[0, 1, 0], angle=rotation[1])
        self.rotate(axis=[0, 0, 1], angle=rotation[2])

        self.view_changed()

    def translate(self, trans=[0,0,0]):
        self.tr.translate(trans)

    def rotate(self, axis=[1,0,0], angle=0):
        self.tr.rotate(axis=axis, angle=angle)

    def view_changed(self):
        self.view.camera.view_changed()

    def render(self):
        return self.canvas.render()

    def reinit_mesh(self, verts, faces, colors):
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])

    def reinit_camera(self, fov):
        self.view.camera.fov = fov
        self.view.camera.view_changed()
