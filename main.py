from ibr.ibr import IBR
from ibr.scene import Scene


ibr = IBR()

data_dir = '/Users/anhtruong/Workspace/view-synthesis-methods/data'
scene_name = '0001'
n_views = 5

scene = Scene()
scene.read_data(data_dir, scene_name, n_views)
scene.add_noise()
scene.denoise()
scene.visualize()
scene.create_mesh()
