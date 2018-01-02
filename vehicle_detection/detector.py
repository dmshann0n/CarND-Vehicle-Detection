import logging as log

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from lane_finder import plotter
from vehicle_detection.features import to_features, SPATIAL_SIZE

class Detector:

    WINDOW_SIZES = [64, 96, 128]
    MIN_Y = 390

    X_OVERLAP = 16
    Y_OVERLAP = [16, 48, 64]

    HEATMAP_THRESHOLD = 4
    HEATMAP_DECAY = 1


    def __init__(self, classifier, show_predicted=False):
        self.classifier = classifier

        self.heatmap = None
        self.frames = 0

        self.show_predicted = show_predicted

    def identify_vehicles(self, img, output_img=None):
        if not output_img:
            output_img = img.copy()

        self._ensure_heatmap_initialized(img)

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

        for bounds in self._get_estimated_positions():
            cv2.rectangle(
                output_img,
                bounds[0],
                bounds[1],
                (0, 0, 255),
                3)

        return output_img

    def _ensure_heatmap_initialized(self, img):
        if self.heatmap is not None:
            return

        self.heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

    def _add_frame(self, vehicle_windows):
        self.frames += 1

        for window in vehicle_windows:
            self.heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        self.heatmap -= self.HEATMAP_DECAY

    def _get_estimated_positions(self):
        thresholded_heatmap = self.heatmap.copy()
        thresholded_heatmap[thresholded_heatmap < self.HEATMAP_THRESHOLD] = 0

        #plotter.Plotter(True).plot_images(np.clip(thresholded_heatmap, 0, 255), cmap='hot')

        found = []
        positions, num_objects = label(thresholded_heatmap)

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

        for window_size, y_overlap in zip(self.WINDOW_SIZES, self.Y_OVERLAP):
            x_position = 0

            while x_position < width:

                yield (
                    (int(x_position), int(y_position)),
                    (int(x_position + window_size), int(y_position + window_size)),
                )

                x_position += self.X_OVERLAP

            y_position += y_overlap
