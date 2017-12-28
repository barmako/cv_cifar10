import cPickle as pickle

import numpy
from sklearn import svm

from BOWPreprocessor import SIFTBOWPreprocessor
from SIFTPreprocessor import SIFTPreprocessor
from cifar_classifier import Classifier

classifier = svm.LinearSVC()

preprocessor = SIFTBOWPreprocessor(n_words=2000,
                                   decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4),
                                                              sift_kp_size=8))
cls = Classifier(
    preprocessor,
    classifier, augment=True)

res = cls.run()

kmeans_model = preprocessor.kmeans_model

to_pickle = [kmeans_model, classifier]

p = 'pickle.p'
pickle.dump(to_pickle, open(p, "wb"), 1)
