import cv2
import time

from ibr.ibr import IBR

depth = cv2.imread('/ipi/scratch/antruong/Middlebury/test (speed)/depth/0030.png')


ibr = IBR()

start = time.time()
ibr.superpixel(depth)
print('Elapsed: ', time.time() - start)
