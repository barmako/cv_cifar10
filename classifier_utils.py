from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# clf = svm.SVC(cache_size=1000, kernel=chi2_kernel)
clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200)

#
# clf = ExtraTreesClassifier()
# clf = svm.LinearSVC()


def init(data, labels):
    clf.fit(data, labels)


def classify(data):
    return clf.predict(data)
