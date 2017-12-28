import cPickle as pickle

import numpy

from BOWPreprocessor import SIFTBOWPreprocessor
from SIFTPreprocessor import SIFTPreprocessor
from cifar_classifier import Classifier


def load():
    p = 'pickle.p'
    from_pickle = pickle.load(open(p, "rb"))

    from_p_pre = SIFTBOWPreprocessor(n_words=300,
                                     decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4),
                                                                sift_kp_size=8))
    from_p_pre.kmeans_model = from_pickle[0]

    from_p_classifier = from_pickle[1]

    return Classifier(from_p_pre, from_p_classifier)
