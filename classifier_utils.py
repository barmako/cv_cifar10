from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier

clf = svm.LinearSVC()

# clf = ExtraTreesClassifier()


def init(data, labels):
    clf.fit(data, labels)


def classify(data):
    return clf.predict(data)
