from utils import Superpixel


class IBR:
    def __init__(self):
        super().__init__()

        self.superpixel_module = Superpixel('SlicAvx2')

    def superpixel(self, image):
        self.superpixel_module.superpixel(image)
