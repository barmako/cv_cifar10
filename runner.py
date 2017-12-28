from cifar_classifier import Classifier
from SIFTPreprocessor import SIFTPreprocessor
from BasePreprocessor import BasePreprocessor
from BOWPreprocessor import SIFTBOWPreprocessor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy

linearSVM = svm.LinearSVC()

base_classifier = Classifier(BasePreprocessor(), linearSVM)


classifier = Classifier(SIFTPreprocessor(), linearSVM)  # 36.89
classifier2 = Classifier(SIFTPreprocessor(8), linearSVM)  # 32.72
classifier3 = Classifier(SIFTBOWPreprocessor(), linearSVM)  # 36.31
classifier4 = Classifier(SIFTPreprocessor(24), linearSVM)  # 36.2

classifier5 = Classifier(SIFTBOWPreprocessor(n_words=400), linearSVM)  # 32.79
classifier6 = Classifier(
    SIFTBOWPreprocessor(n_words=400, decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(4, 31, 4))),
    linearSVM)  # 35.15

classifier7 = Classifier(
    SIFTBOWPreprocessor(n_words=400, decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(4, 31, 4))),
    linearSVM, augment=True)  # 35.75

bow = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(4, 31, 4), sift_kp_size=8)),
    linearSVM)  # 40.08
bow_1 = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(4, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # 40.05

bow2 = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM)  # 41.8

bow3 = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # 42.11

bow4 = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 3), sift_kp_size=8)),
    linearSVM)  # 38.22

bow5 = Classifier(
    SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 3), sift_kp_size=8)),
    linearSVM)


def best_bow(svm):
    return Classifier(
        SIFTBOWPreprocessor(decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
        svm, augment=True)


best_bow_rbf = best_bow(svm.SVC(kernel='rbf'))
best_bow_poly = best_bow(svm.SVC(kernel='poly'))
best_bow_sig = best_bow(svm.SVC(kernel='sigmoid'))

bow_smaller_codebook = Classifier(
    SIFTBOWPreprocessor(n_words=500,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # 39.52

bow_bigger_codebook = Classifier(
    SIFTBOWPreprocessor(n_words=1500,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # 43.14

bow_1500_100 = Classifier(
    SIFTBOWPreprocessor(n_words=1500,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 3), sift_kp_size=8)),
    linearSVM, augment=True)  # 40.46%

bow_2000_100 = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 3), sift_kp_size=8)),
    linearSVM, augment=True)  # 43.42%

bow_huge_codebook = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # 44.28

bow_2500_codebook = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    linearSVM, augment=True)  # ?


knn = KNeighborsClassifier()
bow_2000_64_knn = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    knn, augment=True)  # 37.36%

knn_dis = KNeighborsClassifier(weights='distance')
bow_2000_64_knn_dis = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    knn_dis, augment=True)  # 38.93%

ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200)
bow_2000_ada = Classifier(
    SIFTBOWPreprocessor(n_words=2000,
                        decorated=SIFTPreprocessor(concat=False, kps_idx=numpy.arange(2, 31, 4), sift_kp_size=8)),
    ada, augment=True)
