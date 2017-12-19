import itertools

import cv2
import numpy


class SIFTPreprocessor:
    def __init__(self, sift_kp_size=16, concat=True, kps_idx=numpy.arange(4, 31, 8)):
        self.concat = concat
        self.sift = cv2.SIFT()
        coordinates = list(itertools.product(kps_idx, kps_idx))
        self.kps = [cv2.KeyPoint(i, j, sift_kp_size) for (i, j) in coordinates]

    def preprocess(self, data):
        pass

    def get_descriptors(self, data):
        if self.concat:
            return [self.extract_concated_sift(img) for img in data]
        return [self.extract_sift(img) for img in data]

    def extract_concated_sift(self, img):
        sifts = self.extract_sift(img)
        res = []
        for sift in sifts:
            res.extend(sift)
        return res

    def extract_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.compute(gray, self.kps)
        return des
