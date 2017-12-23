from cifar_classifier import Classifier
from SIFTPreprocessor import SIFTPreprocessor
from BOWPreprocessor import SIFTBOWPreprocessor
from sklearn import svm
import numpy

linearSVM = svm.LinearSVC()

classifier = Classifier(SIFTPreprocessor(), linearSVM)  # 36.89
classifier2 = Classifier(SIFTPreprocessor(8), linearSVM)  # 32.72
classifier3 = Classifier(SIFTBOWPreprocessor(), linearSVM)  # 36.31
classifier4 = Classifier(SIFTPreprocessor(24), linearSVM)  # 36.2

classifier5 = Classifier(SIFTBOWPreprocessor(n_words=400), linearSVM)  # 32.79
classifier6 = Classifier(
    SIFTBOWPreprocessor(n_words=400, decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(4, 31, 4))),
    linearSVM)  # 35.15

classifier7 = Classifier(
    SIFTBOWPreprocessor(n_words=400,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 2), sift_kp_size=4)),
    linearSVM)  #
