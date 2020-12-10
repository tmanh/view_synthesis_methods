import cv2
import time

from ibr.ibr import IBR

depth = cv2.imread('/Users/anhtruong/Downloads/LR/3.5x3.5/114_x[3.5, 3.5]_SR.png')


ibr = IBR()

start = time.time()
ibr.superpixel(depth)
print('Elapsed: ', time.time() - start)
