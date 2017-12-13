import numpy as np


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


pickle_folder = r'cifar-10-batches-py'
data_batches = ['data_batch_{0}'.format(s) for s in range(1, 6)]


def get_pickle_filename(index):
    return data_batches[index]


def get_pickle_path(index):
    if index == -1:
        return pickle_folder + '\\' + 'test_batch'
    return pickle_folder + '\\' + data_batches[index]


def get_pickle(index):
    unpickled = unpickle(get_pickle_path(index))
    datas = unpickled['data']
    labels = unpickled['labels']
    datas = datas.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    labels = np.array(labels)
    return (datas, labels)


def get_test_pickle():
    return get_pickle(-1)
