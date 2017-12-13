import classifier_utils as cu
import dataloader as dl
import descriptors_extractor as de

pickle1 = dl.get_pickle(0)
data = pickle1[0]
labels = pickle1[1]

print "extracting descriptors"
de.initialize(data)
descriptors = de.extract(data)
print "done"

print "svming"
cu.init(descriptors, labels)
print "done"

print "loading test data"
pickle = dl.get_test_pickle()
test_data = pickle[0]
test_labels = pickle[1]

print "classifying test data of size "
print len(test_data)
test_data_desc = de.extract(test_data)
results = cu.classify(test_data_desc)

print "validating results "
errors_count = sum(i != j for i, j in zip(results, test_labels))
print "error count "
print errors_count

print "success rate"
test_data_count = len(test_data)
suc_rate = (test_data_count - errors_count) * 1.0 / test_data_count
print suc_rate * 100
