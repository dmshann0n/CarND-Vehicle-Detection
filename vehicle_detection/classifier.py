""" Abstraction for training the classifier for determing car and not car
    and saving the model for use later """

import glob
import logging as log
import os

import cv2
import numpy as np
import progressbar
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.image import convert_color, load_img

COLOR_SPACE = 'RGB'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
BINS_RANGE = (0, 256)
ORIENTATIONS = 9
PIX_PER_CELL = 8
CELLS_PER_BLOCK = 2
HOG_CHANNEL = None

def dir_to_array(path):
    return glob.glob(path)

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

class Classifier:

    DEFAULT_DATA_PATH = 'data'
    BATCH_SIZE = 500

    CAR_SET = [
        'vehicles/GTI_Far',
        'vehicles/GTI_Left',
        'vehicles/GTI_MiddleClose',
        'vehicles/GTI_Right',
        'vehicles/KITTI_extracted',
    ]

    NOT_CAR_SET = [
        'non-vehicles/GTI',
        'non-vehicles/Extras',
    ]

    def __init__(self):
        self.classifier = LinearSVC()

    def _build_data_path(self, path):
        return os.path.join(self.DEFAULT_DATA_PATH, path, '*')

    def summarize_dataset(self):
        cars, not_cars = self._get_dataset()
        print('Vehicle samples: ', len(cars))
        print('Non-Vehicle samples: ', len(not_cars))

    def _get_dataset(self):
        cars = []
        not_cars = []

        for path in self.CAR_SET:
            cars.extend(dir_to_array(self._build_data_path(path)))

        for path in self.NOT_CAR_SET:
            not_cars.extend(dir_to_array(self._build_data_path(path)))

        return cars, not_cars

    def _extract_features(self, img_paths):
        features = []

        bar = progressbar.ProgressBar()

        for path in bar(img_paths):
            file_features = []
            image = convert_color(load_img(path), COLOR_SPACE)

            for strategy in FEATURE_STRATEGIES:
                file_features.append(strategy(image))

            features.append(np.concatenate(file_features))

        return features

    def train(self):
        log.info('Starting training')

        cars, not_cars = self._get_dataset()

        log.info(f'Extracting vehicles features (set of {len(cars)})')
        car_features = self._extract_features(cars)

        log.info(f'Extracting non-vehicles features (set of {len(not_cars)})')
        not_car_features = self._extract_features(not_cars)

        X = np.vstack((car_features, not_car_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

        log.debug('Shuffling training data')
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2
        )

        log.info('Fitting training data')
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_test, y_test)
        log.info(f'Done fitting. Accuracy of {accuracy:.4f}')

    def predict(self, val):
        return self.classifier.predict(val)

    def to_pickle(self, path):
        # using this as a guide:
        # http://scikit-learn.org/stable/modules/model_persistence.html
        joblib.dump(self.classifier, path)

    def from_pickle(self, path):
        self.classifier = joblib.load(path)
