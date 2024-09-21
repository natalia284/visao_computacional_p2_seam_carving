import cv2
from seam import SeamCarve

img = cv2.imread('caminho/test.png')
mask = cv2.imread('caminho/mascara.png', 0) != 255

sc_img = SeamCarve(img)
sc_img.remove_mask(mask)


cv2.imshow('original', img)
cv2.imshow('removed', sc_img.image())
cv2.waitKey(0)