import classifier_utils as cu
import dataloader as dl
import descriptors_extractor as de

pickle1 = dl.get_pickle(0)
data = pickle1[0]
labels = pickle1[1]


# print "sifting"
# descriptors = []
# for datum in data:
#     sift = seu.extract_sift(datum)
#     if sift is not None:
#         flat_sift = seu.concat_sifts(sift)
#         descriptors.append(np.array(flat_sift))


def flat_image(img):
    flat_img = []
    for row in img:
        for pixel in row:
            for channel in pixel:
                flat_img.append(channel)
    return flat_img


def flat_raw(data):
    flatted_data = []
    for img in data:
        flatted_data.append(flat_image(img))
    return flatted_data


flat_data = flat_raw(data)

print "pcaing"
de.initialize(flat_data)
descriptors = de.extract(flat_data)

print "svming"
cu.init(descriptors, labels)

pickle = dl.get_test_pickle()
test_data = pickle[0]
test_labels = pickle[1]

print "classifying test data of size "
print len(test_data)
test_data_desc = de.extract(flat_raw(test_data))
results = cu.classify(test_data_desc)

print "validating results "
errors_count = sum(i != j for i, j in zip(results, test_labels))
print "error count "
print errors_count

print "success rate"
test_data_count = len(test_data)
suc_rate = (test_data_count - errors_count) * 1.0 / test_data_count
print suc_rate * 100
