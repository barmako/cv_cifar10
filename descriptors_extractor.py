import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import random

import sift_extractor_utils as sift

feature_size = 800
print "BoW dictionary size: %s" % feature_size
pca = PCA(n_components=256)
kmeans = MiniBatchKMeans(n_clusters=feature_size, random_state=0)
print "dictionary clustering done with %s" % kmeans

kmeans_model = None
pca_model = None


def initialize(data):
    global kmeans_model, pca_model
    print "sifting "
    sifts = get_sifts(data)
    print "sift quantization for BOVW dictionary"
    kmeans_train_size = 5000
    rand_sub_sifts = random.sample(sifts, kmeans_train_size)
    kmeans_model = kmeans.fit(rand_sub_sifts)
    # pca_model = pca.fit([get_image_descriptor(image) for image in data])


def get_sifts(data):
    sift_array_array = [sift.extract_sift(img) for img in data]
    sifts = []
    for sift_array in sift_array_array:
        for single_sift in sift_array:
            sifts.append(single_sift)
    return sifts


def extract(data):
    extracted = [get_image_descriptor(image) for image in data]
    # return pca_model.transform(extracted)
    return extracted


def get_image_descriptor(image):
    image_desc = np.zeros(feature_size)
    image_sifts = sift.extract_sift(image)
    words = kmeans_model.predict(image_sifts)
    for word in words:
        image_desc[word] = image_desc[word] + 1
    return image_desc


def __extract_flat_sift(data):
    return [sift.concat_sifts(sift.extract_sift(item)) for item in data]
