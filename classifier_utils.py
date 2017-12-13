from sklearn import svm

from sklearn.decomposition import PCA


def get_classifier(data, labels):
    return fit_svm_model(data, labels)


def fit_svm_model(data, labels):
    clf = svm.SVC()
    return clf.fit(data, labels)


def pca(data, dst_dim):
    pca = PCA(n_components=dst_dim)
    return pca.fit_transform(data)
