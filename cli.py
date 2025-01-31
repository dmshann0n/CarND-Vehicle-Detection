#!/usr/bin/env python3

import logging as log
import os
import random

from clize import run
import cv2
from moviepy.editor import VideoFileClip

from lane_finder import camera_calibration, plotter, lane_finder
from vehicle_detection import classifier, image, detector, features

PLOT = plotter.Plotter(False)
INIT = False

CAMERA_CALIBRATION_PICKLE_PATH = 'camera_calibration.P'
VEHICLE_DETECTION_CLASSIFIER_PATH = 'vehicle_detect.P'

def init_logging():
    log.basicConfig(
        format='%(asctime)s %(message)s',
        level=log.DEBUG)


def get_plotter(show_plots=False):
    PLOT.show_plots = show_plots
    return PLOT


def test_calibrate_camera():
    init_logging()
    plot = get_plotter()

    calibrator = camera_calibration.CameraCalibration(plot)
    calibrator.config()

    PLOT.show_plots = True
    img = cv2.cvtColor(cv2.imread('test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)
    calibrator.undistort(img)


def pickle_camera_calibration(path=None):
    if not path:
        path = CAMERA_CALIBRATION_PICKLE_PATH

    init_logging()
    plot = get_plotter(False)

    calibrator = camera_calibration.CameraCalibration(plot)
    calibrator.config()
    calibrator.to_pickle(path)


def _config_camera_calibrator(show_plots):
    plot = get_plotter()
    calibrator = camera_calibration.CameraCalibration(plot)

    if os.path.exists(CAMERA_CALIBRATION_PICKLE_PATH):
        calibrator.from_pickle(CAMERA_CALIBRATION_PICKLE_PATH)

    return calibrator


def lane_line_single_image(img_path=None, show_plots=False):
    if not img_path:
        img_path = 'test_images/straight_lines1.jpg'

    init_logging()

    calibrator = _config_camera_calibrator(show_plots)
    finder = lane_finder.LaneFinder(calibrator, get_plotter(show_plots))

    img = image.load_img(img_path)
    finder.find_lanes(img)


def vehicle_detect_data_summary():
    clf = classifier.Classifier()
    clf.summarize_dataset()

    cars, not_cars = clf._get_dataset()
    single_car, single_not_car = image.load_img(random.choice(cars)), image.load_img(random.choice(not_cars))

    plotter.Plotter(True).plot_images(single_car, single_not_car)


def vehicle_detect_hog_sample():
    clf = classifier.Classifier()

    cars, not_cars = clf._get_dataset()

    single_car = image.convert_color(image.load_img(random.choice(cars)), features.COLOR_SPACE)
    _, single_car_hog = features.hog_features_by_channel(single_car, 0, visualise=True)
    plotter.Plotter(True).plot_images(single_car, single_car_hog)

    single_not_car = image.convert_color(image.load_img(random.choice(not_cars)), features.COLOR_SPACE)
    _, single_not_car_hog = features.hog_features_by_channel(single_not_car, 0, visualise=True)
    plotter.Plotter(True).plot_images(single_not_car, single_not_car_hog)



def vehicle_detect_train(path=None):
    if not path:
        path = VEHICLE_DETECTION_CLASSIFIER_PATH

    init_logging()

    clf = classifier.Classifier()
    clf.train()
    clf.to_pickle(path)

    log.info(f'Saved classifier to {path}')


def vehicle_detect_single_image(img_path=None):
    if not img_path:
        img_path = 'test_images/test1.jpg'

    init_logging()

    clf = classifier.Classifier()
    clf.from_pickle(VEHICLE_DETECTION_CLASSIFIER_PATH)

    detect = detector.Detector(clf)

    img = image.load_img(img_path)

    calibrator = camera_calibration.CameraCalibration(get_plotter())
    calibrator.config()

    undisorted = calibrator.undistort(img)
    output = detect.identify_vehicles(undisorted)

    plotter.Plotter(True).plot_images(output)


def vehicle_detect_full_run(video_path=None):
    if not video_path:
        video_path = 'test_video.mp4'

    init_logging()

    clf = classifier.Classifier()
    clf.from_pickle(VEHICLE_DETECTION_CLASSIFIER_PATH)

    detect = detector.Detector(clf)

    calibrator = camera_calibration.CameraCalibration(get_plotter())

    log.debug(f'Processing video {video_path}')
    clip = VideoFileClip(video_path)
    updated = clip.fl_image(lambda img: _vehicle_detect_full_run_wrapper(img, detect, calibrator))

    split_path = video_path.split('.')
    new_file = "".join([split_path[0], '_output.', split_path[1]])
    updated.write_videofile(new_file, audio=False)


def _vehicle_detect_full_run_wrapper(img, detect, calibrator):
    undistorted = calibrator.undistort(img)
    return detect.identify_vehicles(undistorted)


def full_run(video_path=None, show_plots=False):
    if not video_path:
        video_path = 'project_video.mp4'

    init_logging()

    calibrator = _config_camera_calibrator(show_plots)
    finder = lane_finder.LaneFinder(calibrator, get_plotter(show_plots))

    clf = classifier.Classifier()
    clf.from_pickle(VEHICLE_DETECTION_CLASSIFIER_PATH)
    detect = detector.Detector(clf)

    log.debug(f'Processing video {video_path}')
    clip = VideoFileClip(video_path)
    updated = clip.fl_image(lambda img: _full_run_wrapper(img, finder, detect, calibrator))

    split_path = video_path.split('.')
    new_file = "".join([split_path[0], '_output.', split_path[1]])
    updated.write_videofile(new_file, audio=False)


def _full_run_wrapper(img, finder, detect, calibrator):
    new_img = finder.find_lanes(img)

    # the lane finder outputs an undistorted (and cropped image) so manage that here..
    undistorrted = calibrator.undistort(img)
    new_img = detect.identify_vehicles(img, output_img=new_img)
    return new_img


if __name__ == '__main__':
    run(full_run,
        test_calibrate_camera,
        pickle_camera_calibration,
        lane_line_single_image,
        vehicle_detect_data_summary,
        vehicle_detect_train,
        vehicle_detect_single_image,
        vehicle_detect_full_run,
        vehicle_detect_hog_sample,
    )
