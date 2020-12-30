from ibr.scene import Scene


data_dir = './data'
scene_name = '0001'
n_views = 5

scene = Scene(600, 90)
scene.read_data(data_dir, scene_name, n_views)
scene.add_noise()
scene.denoise()
scene.create_mesh(scene.extrinsics[0])
scene.visualize()
