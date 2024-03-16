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


def ColorHistogram(image, bins=(8, 8, 8)):

    # Convert BGR to HSV
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the color range
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(HSV, lower, upper)

    # Calculate the histogram
    hist = cv2.calcHist([HSV], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist)

    # print(hist.shape) # (8, 8, 8)

    return hist


# debug
# image = cv2.imread("TIN/n01774384/images/n01774384_4.JPEG")
# image = cv2.resize(image, (256, 256))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# features = HOG(image)
# features = ColorHistogram(image)
# print(features.shape)
# print(features)
