
from cv2 import cv2


frame0 = cv2.imread("./00030.jpg")
base = cv2.imread("./base.jpg")

x_offset = 540 - frame0.shape[1] // 2

base[0:frame0.shape[0], x_offset:x_offset+frame0.shape[1]] = frame0

cv2.imwrite("./merged.jpg", base)



