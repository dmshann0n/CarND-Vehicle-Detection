import logging as log

import cv2
import numpy as np
from skimage.feature import hog

from vehicle_detection.image import convert_color

COLOR_SPACE = 'YCrCb'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
BINS_RANGE = (0, 256)
ORIENTATIONS = 11
PIX_PER_CELL = 8

CELLS_PER_BLOCK = 2

HOG_CHANNEL = [0, 1, 2]
COLOR_HISTOGRAM_CHANNELS = [0, 1, 2]

def hog_features_all(img):
    features = []
    for channel in HOG_CHANNEL:
        features.append(hog_features_by_channel(img, channel))

    return np.ravel(features)

def hog_features_by_channel(img, channel, feature_vector=True, visualise=False):

    return hog(
        img[:, :, channel],
        orientations=ORIENTATIONS,
        pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
        cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK),
        transform_sqrt=True,
        visualise=visualise,
        feature_vector=feature_vector,
        block_norm='L2-Hys',
    )

def spatial_features(img):
    return cv2.resize(img, SPATIAL_SIZE).ravel()

def color_histogram_features(img):
    histograms = []
    for channel in COLOR_HISTOGRAM_CHANNELS:
        histograms.append(np.histogram(
            img[:, :, channel],
            bins=HIST_BINS,
            range=BINS_RANGE)[0]
        )

    return np.concatenate(histograms)


def to_features(img):
    img = convert_color(img, COLOR_SPACE)

    features = []

    features.append(hog_features_all(img))
    features.append(spatial_features(img))
    features.append(color_histogram_features(img))

    return np.concatenate(features)
