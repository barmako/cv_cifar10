import cv2

sift = cv2.SIFT()
kps = [cv2.KeyPoint(7, 7, 1), cv2.KeyPoint(7, 15, 1), cv2.KeyPoint(15, 7, 1), cv2.KeyPoint(15, 15, 1)]


# dense-sift
def extract_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.compute(gray, kps)
    return des


def concat_sifts(sift):
    return [item for sublist in sift for item in sublist]
