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
    return pickle_folder + '\\' + data_batches[index]


def get_pickle(index):
    return unpickle(get_pickle_path(index))
