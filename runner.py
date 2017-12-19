from cifar_classifier import Classifier
from SIFTPreprocessor import SIFTPreprocessor
from BOWPreprocessor import BOWPreprocessor
from sklearn import svm

linearSVM = svm.LinearSVC()

classifier = Classifier(SIFTPreprocessor(), linearSVM)

