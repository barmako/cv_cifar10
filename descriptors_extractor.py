import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import sift_extractor_utils as sift

feature_size = 1000
pca = PCA(n_components=256)
kmeans = KMeans(n_clusters=feature_size, random_state=0)

kmeans_model = None
pca_model = None


def initialize(data):
    global kmeans_model, pca_model
    sifts = get_sifts(data)
    kmeans_model = kmeans.fit(sifts)
    pca_model = pca.fit([get_image_descriptor(image) for image in data])


def get_sifts(data):
    sift_array_array = [sift.extract_sift(img) for img in data]
    sifts = []
    for sift_array in sift_array_array:
        for single_sift in sift_array:
            sifts.append(single_sift)
    return sifts


def extract(data):
    extracted = [get_image_descriptor(image) for image in data]
    return pca_model.transform(extracted)


def get_image_descriptor(image):
    image_desc = np.zeros(feature_size)
    image_sifts = sift.extract_sift(image)
    words = kmeans_model.predict(image_sifts)
    for word in words:
        image_desc[word] = image_desc[word] + 1
    return image_desc


def __extract_flat_sift(data):
    return [sift.concat_sifts(sift.extract_sift(item)) for item in data]
