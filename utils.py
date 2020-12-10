from fast_slic import Slic
from fast_slic.avx2 import SlicAvx2
from skimage.segmentation import slic as skislic


class Superpixel:
    def __init__(self, mode, cfg=None):
        super().__init__()

        if mode == 'FastSlic':
            self.generator = FastSlic(cfg)
        elif mode == 'SkiSlic':
            self.generator = SkiSlic(cfg)
        elif mode == 'SlicAvx2':
            self.generator = FastSlicAvx2(cfg)

    def superpixel(self, image):
        self.generator.superpixel(image)
        

class FastSlic:
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {'num_components': 1600, 'compactness': 10}

        self.slic = Slic(num_components=cfg['num_components'], compactness=cfg['compactness'])

    @staticmethod
    def info():
        print('Source: ', 'https://github.com/Algy/fast-slic')

    def superpixel(self, image):
        return self.slic.iterate(image)


class FastSlicAvx2:
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {'num_components': 1600, 'compactness': 10}

        self.slic = SlicAvx2(num_components=cfg['num_components'], compactness=cfg['compactness'])

    @staticmethod
    def info():
        print('Source: ', 'https://github.com/Algy/fast-slic')

    def superpixel(self, image):
        return self.slic.iterate(image)


class SkiSlic:
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {'num_components': 1600, 'compactness': 10}
        
        self.cfg = cfg

    @staticmethod
    def info():
        print('Source: ', 'https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic')

    def superpixel(self, image):
        segments = skislic(image, n_segments=self.cfg['num_components'], compactness=self.cfg['compactness'], start_label=0)
        print(segments)
