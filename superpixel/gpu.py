from modules.gSLICrPy.gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr


class GpuSlic:
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {'num_components': 1600, 'compactness': 10}

        self.__CUDA_gSLICr__ = __get_CUDA_gSLICr__()

    @staticmethod
    def info():
        print('Source: ', 'https://github.com/mikigom/gSLICrPy')

    def superpixel(self, image):
        img_size_y, img_size_x = image.shape[0:2]

        out = CUDA_gSLICr(self.__CUDA_gSLICr__,
                image,
                img_size_x=img_size_x,
                img_size_y=img_size_y,
                n_segs=10,
                spixel_size=20,
                coh_weight=0.6,
                n_iters=50,
                color_space=2,
                segment_color_space=2,
                segment_by_size=True,
                enforce_connectivity=True,
                out_name='example_results')
        return out
