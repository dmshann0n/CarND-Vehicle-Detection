import logging as log

import cv2
import matplotlib.pyplot as plt
import numpy as np

WIDTH = HEIGHT = 5

class Plotter():
    def __init__(self, show_plots):
        self.show_plots = show_plots

    def plot_chart(self, data):
        if not self.show_plots:
            return

        plt.plot(data)
        plt.show()

    def plot_pair(self, img0, img1, cmaps):
        f, axes = plt.subplots(1, 2, figsize=(WIDTH * 2, HEIGHT))
        axes[0].imshow(img0, cmap=cmaps[0])
        axes[1].imshow(img1, cmap=cmaps[1])
        plt.show()

    def plot_three(self, img0, img1, img2, cmaps):
        f, axes = plt.subplots(1, 3, figsize=(WIDTH * 3, HEIGHT))
        axes[0].imshow(img0, cmap=cmaps[0])
        axes[1].imshow(img1, cmap=cmaps[1])
        axes[2].imshow(img2, cmap=cmaps[2])
        plt.show()

    def plot_images(self, *imgs, **kwargs):
        if not self.show_plots:
            return

        cmap = None
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']

        f, axes = plt.subplots(len(imgs), 1, figsize=(WIDTH, HEIGHT * len(imgs)))
        f.tight_layout()

        # wrap the axes in a list so I don't have to
        # special case below
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for i, img in enumerate(imgs):
            if callable(img):
                axes[i].imshow(img(), cmap=cmap)
            else:
                axes[i].imshow(img, cmap=cmap)

        plt.show()
