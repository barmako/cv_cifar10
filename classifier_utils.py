from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

clf = svm.SVC(cache_size=1000, kernel=chi2_kernel)

# clf = ExtraTreesClassifier()
# clf = svm.LinearSVC()


def init(data, labels):
    clf.fit(data, labels)


def classify(data):
    return clf.predict(data)
