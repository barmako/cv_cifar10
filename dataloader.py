import numpy as np
import cPickle


def unpickle(file):
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


def get_pickle_by_index(index):
    path = get_pickle_path(index)
    return get_pickle_by_path(path)


def get_pickle_by_path(param):
    unpickled = unpickle(param)
    data = unpickled['data']
    labels = unpickled['labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    labels = np.array(labels)
    return data, labels


def get_pickle(param):
    if isinstance(param, int):
        return get_pickle_by_index(param)
    return get_pickle_by_path(param)


def get_all():
    res_data = []
    res_labels = []
    for i in range(0, 5):
        pickle = get_pickle(i)
        data = pickle[0]
        res_data.extend(data)
        label = pickle[1]
        res_labels.extend(label)
    return [res_data, res_labels]


def get_test_pickle():
    return get_pickle(-1)
