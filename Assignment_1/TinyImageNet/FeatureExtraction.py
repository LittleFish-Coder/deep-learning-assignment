import cv2
import numpy as np


def HOG(image):

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # channel 3 to 1

    # HOG descriptor
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(
        winSize,
        blockSize,
        blockStride,
        cellSize,
        nbins,
        derivAperture,
        winSigma,
        histogramNormType,
        L2HysThreshold,
        gammaCorrection,
        nlevels,
    )

    features = hog.compute(image)

    return features


# debug
# image = cv2.imread("TIN/n01774384/images/n01774384_4.JPEG")
# image = cv2.resize(image, (256, 256))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# features = HOG(image)
# print(features.shape)
# print(features)
