import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class BOWPreprocessor:
    def __init__(self, decorated, n_words=1000, kmeans_train_size=5000):
        self.kmeans_train_size = kmeans_train_size
        self.n_words = n_words
        self.decorated = decorated
        self.kmeans = MiniBatchKMeans(n_words)
        self.kmeans_model = None

    def preprocess(self, data):
        self.decorated.preprocess(data)
        base_descs = self.decorated.get_descriptors(data)
        kmeans_train_data = random.sample(base_descs, self.kmeans_train_size)
        self.kmeans_model = kmeans_train_data.fit(kmeans_train_data)

    def get_descriptors(self, data):
        base_descriptors = self.decorated.get_descriptors(data)
        return [self.quantize_descriptor(b_des) for b_des in base_descriptors]

    def quantize_descriptor(self, base_descriptor):
        image_desc = np.zeros(self.n_words)
        words = self.kmeans_model.predict(base_descriptor)
        for word in words:
            image_desc[word] = image_desc[word] + 1
        return image_desc
