import itertools

import cv2
import numpy

sift = cv2.SIFT()
colIndx = numpy.arange(4, 31, 8)
rowIndx = numpy.arange(4, 31, 8)
coordinates = list(itertools.product(colIndx, rowIndx))
kps = [cv2.KeyPoint(i, j, 16) for (i, j) in coordinates]


# dense-sift
def extract_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.compute(gray, kps)
    return des


def concat_sifts(sift):
    return [item for sublist in sift for item in sublist]

def display_kps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kp = sift.detect(gray, None)
    img2 = cv2.drawKeypoints(gray, kps)
    cv2.imwrite('sift_keypoints_before.jpg', img)
    cv2.imwrite('sift_keypoints.jpg', img2)


# import dataloader
#
# pickle = dataloader.get_pickle(0)
# data = pickle[0]
# datum = data[0]
#
# display_kps(datum)
