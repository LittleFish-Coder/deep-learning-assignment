import cv2
import numpy as np


def HOG(image):
    """
    Extract HOG features from an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Flattened HOG feature vector.
    """

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # channel 3 to 1

    # HOG parameters
    winSize = image.shape
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    # Create a HOG descriptor object
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Compute the HOG features
    features = hog.compute(image).flatten()

    return features


def ColorHistogram(image, bins=(8, 8, 8)):
    """
    Extract color histogram features from an image.

    Args:
        image (numpy.ndarray): Input image.
        bins (tuple): Number of bins for each color channel.

    Returns:
        numpy.ndarray: Color histogram feature vector.
    """

    # Convert BGR to HSV
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram
    hist = cv2.calcHist([HSV], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()  # flatten the histogram (8, 8, 8) -> (512, )

    return hist


def SIFT(image):
    """
    Extract SIFT features from an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        tuple: (keypoints, descriptors)
            keypoints (list): List of detected keypoints.
            descriptors (numpy.ndarray): Array of SIFT descriptors.
    """

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT descriptor
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


# normalize the feature to a fiexd dimension: bag-of-words
def BOW(feature):
    # feature: a numpy array of shape number_of_keypoints * 128
    # return a numpy array of shape 1 * 128

    return np.mean(feature, axis=0)


# debug
# image = cv2.imread("TIN/n01774384/images/n01774384_5.JPEG")
# image = cv2.resize(image, (32, 32))
# features = HOG(image)
# print(features.shape) # (324,)
# features = ColorHistogram(image)
# print(features.shape) # (512,)
# print(features)
# keypoints, descriptors = SIFT(image)
# print(descriptors.shape)
# print(descriptors[0])
# print(keypoints)
# print(len(keypoints))
