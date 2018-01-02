import logging as log

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from lane_finder import plotter
from vehicle_detection.features import to_features, SPATIAL_SIZE

class Heatmap:
    def __init__(self, length, threshold):
        self.frames = []
        self.max_length = length
        self.threshold = threshold

    def add_frame(self, windows):
        self.frames.append(windows)
        if len(self.frames) > self.max_length:
            self.frames.pop(0)

    def get_heatmap(self, img):
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

        for frame in self.frames:
            for window in frame:
                heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        heatmap[heatmap < self.threshold] = 0

        return heatmap


class Detector:

    WINDOW_SIZES = [48, 72, 72, 96]
    MIN_Y = 340
    LEFT_X = 256 # hack!
    X_OVERLAP = [16, 36, 48, 48]
    Y_OVERLAP = [24, 72, 72, 0]

    def __init__(self, classifier, show_predicted=True):
        self.classifier = classifier
        self.heatmap = Heatmap(20, 19)
        self.show_predicted = show_predicted

    def identify_vehicles(self, img, output_img=None):
        if output_img is None:
            output_img = img.copy()

        vehicle_windows = []

        # sliding window to find all possible matches
        for window in self._generate_windows(img):

            cropped = cv2.resize(
                img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                (64, 64))

            is_car = self.classifier.predict(to_features(cropped)) == 1

            if is_car:
                vehicle_windows.append(window)

            if self.show_predicted:
                cv2.rectangle(
                    output_img,
                    window[0],
                    window[1],
                    (0, 255, 0) if is_car else (255, 0, 0),
                    3 if is_car else 1)

        self._add_frame(vehicle_windows)

        for bounds in self._get_estimated_positions(img):
            cv2.rectangle(
                output_img,
                bounds[0],
                bounds[1],
                (0, 0, 255),
                3)

        return output_img

    def _add_frame(self, vehicle_windows):
        self.heatmap.add_frame(vehicle_windows)

    def _get_estimated_positions(self, img):

        heatmap = self.heatmap.get_heatmap(img)

        found = []
        positions, num_objects = label(heatmap)

        #plotter.Plotter(True).plot_three(img, heatmap, positions, [None, 'hot', None])

        for ordinal in range(1, num_objects + 1):
            nonzero_y, nonzero_x = (positions == ordinal).nonzero()
            bounds = (
                (np.min(nonzero_x), np.min(nonzero_y)),
                (np.max(nonzero_x), np.max(nonzero_y))
            )

            found.append(bounds)

        return found

    def _generate_windows(self, img):
        width, _ = img.shape[1], img.shape[0]

        y_position = self.MIN_Y

        for window_size, y_overlap, x_overlap in zip(self.WINDOW_SIZES, self.Y_OVERLAP, self.X_OVERLAP):
            x_position = self.LEFT_X

            while x_position < width:

                yield (
                    (int(x_position), int(y_position)),
                    (int(x_position + window_size), int(y_position + window_size)),
                )

                x_position += x_overlap

            y_position += y_overlap
