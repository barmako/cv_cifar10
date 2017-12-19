import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from SIFTPreprocessor import SIFTPreprocessor


class SIFTBOWPreprocessor:
    def __init__(self, decorated=SIFTPreprocessor(concat=False), n_words=1000, kmeans_train_size=5000):
        self.kmeans_train_size = kmeans_train_size
        self.n_words = n_words
        self.decorated = decorated
        self.kmeans = MiniBatchKMeans(n_words)
        self.kmeans_model = None

    def preprocess(self, data):
        self.decorated.preprocess(data)
        base_descs = self.decorated.get_descriptors(data)
        sifts = []
        for sifts_array in base_descs:
            for sift in sifts_array:
                sifts.append(sift)

        print "Clustering sifts (shape %s) with k-means to build a codebook" % str(np.shape(sifts))
        kmeans_train_data = random.sample(sifts, self.kmeans_train_size)
        self.kmeans_model = self.kmeans.fit(kmeans_train_data)

    def get_descriptors(self, data):
        data_sifts = self.decorated.get_descriptors(data)
        return [self.quantize_descriptor(sifts) for sifts in data_sifts]

    def quantize_descriptor(self, sifts):
        image_desc = np.zeros(self.n_words)
        words = self.kmeans_model.predict(sifts)
        for word in words:
            image_desc[word] = image_desc[word] + 1
        return image_desc
