from sklearn.decomposition import PCA

import sift_extractor_utils as sift

pca = PCA(n_components=128)


def initialize(data):
    sifted = [sift.concat_sifts(sift.extract_sift(item)) for item in data]
    pca.fit(sifted)


def extract(data):
    sifted = [sift.concat_sifts(sift.extract_sift(item)) for item in data]
    return pca.transform(sifted)
