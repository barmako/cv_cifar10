import numpy as np


class BasePreprocessor:
    def __init__(self):
        pass

    def preprocess(self, data):
        pass

    def get_descriptors(self, data):
        return np.reshape(data, (len(data), 32 * 32 * 3))