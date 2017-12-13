from sklearn import svm

clf = svm.LinearSVC()


def init(data, labels):
    clf.fit(data, labels)


def classify(data):
    return clf.predict(data)


def get_classifier(data, labels):
    return fit_svm_model(data, labels)


def fit_svm_model(data, labels):
    clf = svm.SVC()
    return clf.fit(data, labels)
