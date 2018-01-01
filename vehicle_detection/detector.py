import cv2
import numpy as np

from vehicle_detection.features import to_features, SPATIAL_SIZE
from lane_finder import plotter

class Detector:

    START_Y = 390
    START_WINDOW_SIZE = 16
    WINDOW_GROWTH = 32
    OVERLAP = 0.5
    PIX_PER_CELL = 8


    def __init__(self, classifier):
        self.classifier = classifier

    def draw_bounds(self, img, output_img=None):
        if not output_img:
            output_img = img.copy()

        # extract hog features from entire search region (START_Y to height, all X)
        plot = plotter.Plotter(True)

        # sliding window to find all possible matches
        for window in self._generate_windows(img):


            cropped = cv2.resize(
                img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                (64, 64))

            is_car = self.classifier.predict(to_features(cropped)) == 1

            cv2.rectangle(
                output_img,
                window[0],
                window[1],
                (0, 255, 0) if is_car else (255, 0, 0),
                3 if is_car else 1)

        plot.plot_images(output_img)

    def _generate_windows(self, img):
        width, height = img.shape[1], img.shape[0]

        window_size = self.START_WINDOW_SIZE
        y = self.START_Y

        while y + window_size < height:
            x = 0

            while x < width:
                yield (
                    (int(x), int(y)),
                    (int(x + window_size), int(y + window_size)),
                )

                x = x + window_size * self.OVERLAP

            y = y + window_size * self.OVERLAP
            window_size = window_size + self.WINDOW_GROWTH
