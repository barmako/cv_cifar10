import cv2

hog_descriptor = cv2.HOGDescriptor()


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h = hog_descriptor.compute(gray)
    return h


def test():
    import dataloader

    pickle = dataloader.get_pickle(0)
    data = pickle[0]
    datum = data[0]

    return extract(datum)
