""" Module containing functions managing loading/manipulating images """

import cv2

FROM_RGB_COLOR_SPACE_MAP = {
    'HSV': cv2.COLOR_RGB2HSV,
    'LUV': cv2.COLOR_RGB2LUV,
    'HLS': cv2.COLOR_RGB2HLS,
    'YUV': cv2.COLOR_RGB2YUV,
    'YCrCb': cv2.COLOR_RGB2YCrCb
}

def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def convert_color(img, to_space):
    if to_space in FROM_RGB_COLOR_SPACE_MAP:
        return cv2.cvtColor(img, FROM_RGB_COLOR_SPACE_MAP[to_space])

    return img.copy()