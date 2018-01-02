""" Abstraction for training the classifier for determing car and not car
    and saving the model for use later """

import glob
import logging as log
import os

import cv2
import numpy as np
import progressbar
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.image import load_img
from vehicle_detection.features import to_features

def dir_to_array(path):
    return glob.glob(path)

class Classifier:

    DEFAULT_DATA_PATH = 'data'

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
        self.scaler = StandardScaler()

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
            image = load_img(path)
            features.append(to_features(image))

        return features

    def train(self):
        log.info('Starting training')

        cars, not_cars = self._get_dataset()

        log.info(f'Extracting vehicles features (set of {len(cars)})')
        car_features = self._extract_features(cars)

        log.info(f'Extracting non-vehicles features (set of {len(not_cars)})')
        not_car_features = self._extract_features(not_cars)

        X = np.vstack((car_features, not_car_features)).astype(np.float64)

        self.scaler.fit(X)
        scaled_X = self.scaler.transform(X)

        y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

        log.debug('Shuffling training data')
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2
        )

        log.info('Fitting training data')
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_test, y_test)
        log.info(f'Done fitting. Accuracy of {accuracy:.4f}')

    def predict(self, features):
        single = np.array(features).reshape(1, -1)
        return self.classifier.predict(self.scaler.transform(single))

    def to_pickle(self, path):
        # using this as a guide:
        # http://scikit-learn.org/stable/modules/model_persistence.html
        joblib.dump({
            'model': self.classifier,
            'scaler': self.scaler }, path)

    def from_pickle(self, path):
        obj = joblib.load(path)
        self.classifier = obj['model']
        self.scaler = obj['scaler']
