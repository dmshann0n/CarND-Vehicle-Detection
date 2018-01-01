import cv2
import numpy as np
from skimage.feature import hog

from vehicle_detection.image import convert_color

COLOR_SPACE = 'RGB'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
BINS_RANGE = (0, 256)
ORIENTATIONS = 9
PIX_PER_CELL = 8
CELLS_PER_BLOCK = 2
HOG_CHANNEL = None


def hog_features_all(img):
    features = []
    for channel in range(img.shape[2]):
        features.append(hog_features_by_channel(img, channel))

    return np.ravel(features)


def hog_features_by_channel(img, channel=0):
    return hog(
        img[:, :, channel],
        orientations=ORIENTATIONS,
        pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
        cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK),
        transform_sqrt=True,
        feature_vector=True,
        block_norm='L2-Hys',
    )

def spatial_features(img):
    return cv2.resize(img, SPATIAL_SIZE).ravel()

def color_histogram_features(img):
    histograms = []
    for channel in [0, 1, 2]:
        histograms.append(np.histogram(
            img[:, :, channel],
            bins=HIST_BINS,
            range=BINS_RANGE)[0]
        )

    return np.concatenate(histograms)

FEATURE_STRATEGIES = [
    hog_features_all,
    spatial_features,
    color_histogram_features
]

def to_features(img):
    img = convert_color(img, COLOR_SPACE)

    features = []
    for strategy in FEATURE_STRATEGIES:
        features.append(strategy(img))
    return np.concatenate(features)
