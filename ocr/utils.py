# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ocr.cropimage import trimmed_image, pad_and_resize


def default_background_threshold(img):
    """
    Get the background threshold for a grayscale image as the average pixel
    color minus 2*the std of the pixel color.

    If this value is negative, 0 is returned instead.
    """
    assert len(img.shape) == 2
    ravel = img.ravel()
    avg = np.mean(ravel)
    std = np.std(ravel)
    bg_thresh = max(0, avg - 2*std)
    return bg_thresh


def grayscale_to_black_and_white(img, thresh=None):
    """
    Convert a grayscale image to just black and white pixels.

    The default_background_threshold() is used if thresh is None.
    """
    if thresh is None:
        thresh = default_background_threshold(img)
    return np.array([[0 if pix <= thresh else 255 for pix in row] for row in img]).astype(np.uint8)


def resize(img, width, height, thresh):
    """
    Smart resize on a character by maintaining aspect ratio.

    Args:
        img (numpy.ndarray): Image of dark character on light background.
        width (int): Desired resize width.
        height (int): Desired resize height.
        thresh (int): Background threshold up to which, a pixel value is
            considered colored/is containing dark text.

    Returns:
        numpy.ndarray: The resized image.
    """
    trimmed = trimmed_image(img, thresh=thresh)
    img = pad_and_resize(trimmed, width, height, bg=255)
    return img


def img_to_vector(img):
    """
    Convert a 2D image to a 1D feature vector.

    TODO: Create a FeatureVector class that the classifiers will take for
    experimenting with different features as opposed to just implementing
    this function.

    Args:
        img (numpy.ndarray): Image to convert.

    Returns:
        numpy.ndarray: 1D feature vector.
    """
    return np.reshape(img, (img.shape[0] * img.shape[1], ))


def transform(img, width, height, thresh):
    """Wrapper for resizing and vectorizing an image."""
    img = grayscale_to_black_and_white(img, thresh)
    img = resize(img, width, height, thresh)
    vec = img_to_vector(img)
    return vec


def split_data(X, y, retain, classes):
    """Split the data set into training and test data."""
    test_X = []
    test_y = []
    train_X = []
    train_y = []
    for i in xrange(classes):
        start_idx = len(y) / classes * i
        split_idx = int(start_idx + len(y) / classes * retain)
        end_idx = len(y) / classes * (i + 1)
        train_X += X[start_idx:split_idx]
        train_y += y[start_idx:split_idx]
        test_X += X[split_idx:end_idx]
        test_y += y[split_idx:end_idx]
    return (np.array(train_X), np.array(train_y), np.array(test_X),
            np.array(test_y))


def show_img(img, cmap="gray"):
    """Just display an image."""
    fig = plt.figure()
    plt.imshow(img, cmap=cmap)

    # Hide tick values on X and Y axis
    plt.xticks([])
    plt.yticks([])

    fig.show()

    # Keep image alive by waiting for user input
    raw_input("Press 'Enter' to close the image.")
    plt.close(fig)

