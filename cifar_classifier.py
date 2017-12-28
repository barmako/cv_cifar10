# import cPickle
import pickle
import random
import numpy as np
from sklearn.metrics import confusion_matrix

import dataloader as dl
from augmentator import Augmentator


class Classifier:
    def __init__(self, preprocessor, classifier, augment=False, subsample_for_learning=None):
        self.subsample_for_learning = subsample_for_learning
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.augment = augment

    def run(self):
        print "Running classifier"
        raw_data, labels = self.load_training_data()
        if self.subsample_for_learning is not None:
            zipped = zip(raw_data, labels)
            random_sample = random.sample(zipped, self.subsample_for_learning)
            raw_data, labels = zip(*random_sample)
            raw_data = np.asarray(raw_data)
            labels = np.asarray(labels)
        if self.augment:
            print "Augmenting data"
            aug_data, new_labels = Augmentator().flip_augment(raw_data, labels)
            raw_data.extend(aug_data)
            labels.extend(new_labels)

        print "Preprocessing data"
        self.preprocessor.preprocess(raw_data)
        descriptors = self.preprocessor.get_descriptors(raw_data)
        print "Example descriptor- %s" % random.sample(descriptors, 1)

        print "Training classifier"
        self.classifier.fit(descriptors, labels)

        test_data, test_labels = self.load_test_data()

        return self.predict(test_data, test_labels)

    def predict(self, test_data, test_labels):
        print "Classifying test data"
        test_descriptors = self.preprocessor.get_descriptors(test_data)
        results = self.classifier.predict(test_descriptors)

        print "Validating results"
        suc_rate = self.calc_success_rate(results, test_data, test_labels)

        print "confusion matrix"
        conf_matrix = confusion_matrix(test_labels, results)
        print conf_matrix
        return results, conf_matrix, suc_rate

    def predict_pickle(self, pickle_path):
        data = dl.get_pickle_by_path(pickle_path)
        return self.predict(data[0], data[1])

    def calc_success_rate(self, results, test_data, test_labels):
        errors_count = sum(i != j for i, j in zip(results, test_labels))
        print "error count "
        print errors_count
        print "success rate"
        test_data_count = len(test_data)
        suc_rate = (test_data_count - errors_count) * 1.0 / test_data_count
        print suc_rate * 100
        return suc_rate

    def load_training_data(self):
        print "loading train data"
        pickle = dl.get_all()
        data = pickle[0]
        labels = pickle[1]
        print "# train samples: %s" % len(data)
        return data, labels

    def load_test_data(self):
        print "loading test data"
        pickle = dl.get_test_pickle()
        test_data = pickle[0]
        test_labels = pickle[1]
        print "test data size: %s" % len(test_data)
        return test_data, test_labels

    def save(self, path='trained_model.p'):
        pickle.dump(self, open(path, "wb"))


def load(pickle_path):
    return pickle.load(open(pickle_path, "rb"))
