import cv2


# This can be replaced with dense-sift
def extract_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)
    # kp, des = sift.compute(gray, kp)
    kp, des = sift.detectAndCompute(gray, None)
    return des


def concat_sifts(sift):
    return [item for sublist in sift for item in sublist]
