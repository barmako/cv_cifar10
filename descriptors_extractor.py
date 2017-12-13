from sklearn.decomposition import PCA

pca = PCA(n_components=128)


def initialize(data):
    pca.fit(data)

def extract(data):
    return pca.transform(data)
