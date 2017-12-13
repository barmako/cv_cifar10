import classifier_utils as cu
import dataloader as dl

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


flat_data = []
for img in data:
    flat_data.append(flat_image(img))

print "pcaing"
descriptors = cu.pca(flat_data, 128)

print "svming"
classifier = cu.get_classifier(descriptors, labels)
print classifier
