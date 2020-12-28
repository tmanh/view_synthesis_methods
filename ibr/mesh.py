import numpy as np

from vispy import scene, gloo
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
from vispy.gloo.wrappers import read_pixels

class ViewRenderer():
    def __init__(self, fov, resolution, verts, faces, colors, translation, rotation, near, far):
        self.canvas = Canvas(size=(resolution, resolution))
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'perspective'
        self.view.camera.fov = fov
        
        self.mesh = visuals.Mesh(shading=None)
        self.mesh.attach(Alpha(1.0))

        self.view.add(self.mesh)
        
        self.tr = self.view.camera.transform

        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])

        self.near = near
        self.far = far
        
        self.transform(translation, rotation)

        self.view_changed()

    def translate(self, trans=[0,0,0]):
        self.tr.translate(trans)

    def rotate(self, axis=[1,0,0], angle=0):
        self.tr.rotate(axis=axis, angle=angle)

    def transform(self, translation, rotation):
        self.translate(translation)

        self.rotate(axis=[1, 0, 0], angle=rotation[0])
        self.rotate(axis=[0, 1, 0], angle=rotation[1])
        self.rotate(axis=[0, 0, 1], angle=rotation[2])

        self.view_changed()

    def view_changed(self):
        self.view.camera.view_changed()

    def render(self):
        color, depth = self.canvas.render()

        depth = depth * 2 - 1
        depth = (2 * self.near * self.far) / (self.far + self.near - depth * (self.far - self.near)) / 5

        return color, depth


class Canvas(scene.SceneCanvas):
    def __init__(self, title='VisPy canvas', size=(800, 600), position=None,
                 show=False, autoswap=True, app=None, create_native=True,
                 vsync=False, resizable=True, decorate=True, fullscreen=False,
                 config=None, shared=None, keys=None, parent=None, dpi=None,
                 always_on_top=False, px_scale=1, bgcolor='black'):
        super(Canvas, self).__init__(title=title, size=size, position=position,
                 show=show, autoswap=autoswap, app=app, create_native=create_native,
                 vsync=vsync, resizable=resizable, decorate=decorate, fullscreen=fullscreen,
                 config=config, shared=shared, keys=keys, parent=parent, dpi=dpi,
                 always_on_top=always_on_top, px_scale=px_scale, bgcolor=bgcolor)

        gloo.set_state(depth_test=True)

    def render(self, region=None, size=None, bgcolor=None, crop=None):
        self.set_current()
        # Set up a framebuffer to render to
        offset = (0, 0) if region is None else region[:2]
        csize = self.size if region is None else region[2:]
        s = self.pixel_scale

        size = tuple([x * s for x in csize]) if size is None else size
        
        fbo = gloo.FrameBuffer(color=gloo.RenderBuffer(size[::-1]),
                               depth=gloo.RenderBuffer(size[::-1]))

        self.push_fbo(fbo, offset, csize)

        try:
            if crop is None:
                h, w = fbo.color_buffer.shape[:2]
                crop = (0, 0, w, h)

            self._draw_scene(bgcolor=bgcolor)

            color = read_pixels(crop, mode='color', out_type='unsigned_byte')
            depth = read_pixels(crop, mode='depth', out_type='float')

            return color, depth
        finally:
            self.pop_fbo()
