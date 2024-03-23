import cv2
import numpy as np
from sklearn.cluster import KMeans


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


def build_vocabulary(descriptors, k=100):
    """
    Build a visual vocabulary using k-means clustering on SIFT descriptors.

    Args:
        descriptors (list): List of SIFT descriptors from all training images.
        k (int): Number of visual words (cluster centers).

    Returns:
        numpy.ndarray: Visual words (cluster centers).
    """
    descriptors_array = np.vstack(descriptors)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(descriptors_array)
    return kmeans


def bow_encoding(descriptors, visual_words):
    """
    Perform bag-of-words encoding using SIFT descriptors and visual words.

    Args:
        descriptors (numpy.ndarray): SIFT descriptors from an image.
        visual_words (numpy.ndarray): Visual words (cluster centers).

    Returns:
        numpy.ndarray: Bag-of-words histogram.
    """
    visual_words.predict(descriptors)
    histogram = np.bincount(visual_words.labels_, minlength=visual_words.n_clusters)
    return histogram


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
