#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create and test a K Nearest Neighbors Classifier.
"""

from __future__ import print_function

import sys
import os
import cv2
import numpy as np
import pickle
import string
import logging

from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from ocr.utils import transform, split_data, show_img, grayscale_to_black_and_white, resize
from ocr.classifiers.base import Classifier

LOGGER = logging.getLogger(__name__)
ALPHA_NUMERIC = string.digits + string.ascii_uppercase + string.ascii_lowercase


class KNN(Classifier):
    """K Nearest Neighbors Classifier."""

    def predictions_and_metadata(self, data, k=5):
        labels = self.training_labels()
        distances, indeces = self.classifier().kneighbors(data, n_neighbors=k)
        label_pred = [[labels[i] for i in idxs] for idxs in indeces]
        predictions = [max(idxs, key=idxs.count) for idxs in label_pred]
        return predictions, label_pred, distances

    def test(self, data, expected, **kwargs):
        predictions, all_predictions, _ = self.predictions_and_metadata(data, **kwargs)
        correct = 0
        for i, x in enumerate(expected):
            pred = predictions[i]
            LOGGER.debug("expected: {}, guess: {}, guesses: {}"
                         .format(x, pred, all_predictions[i]))
            if x == pred:
                correct += 1
        print(correct * 100.0 / len(expected), "%")


def load_shrinked_imgs(dirname, width=20, samples=55, classes=62, retain=0.8,
                       colors=256):
    """
    Same as load_english_hand, but for an already shrinked data set that does
    not need to be resized.

    Args:
        dirname (str): Directory containing sample data.
        width (Optional[int]): Width and height of each sample.
            Defaults to 20.
        samples (Optional[int]): Number of samples for each class.
        classes (Optional[int]): Number of classes. Defaults to 62.
        retain (Optional[float]): Percentage of sample data to retain as
            training data. The rest is used as test data. Defaults to 0.8.
        colors (Optional[int]): Number of colors. Defaults to 256.

    Returns:
        numpy.ndarray: Training data.
        numpy.ndarray: Training labels.
        numpy.ndarray: Test data.
        numpy.ndarray: Test labels.
    """
    X = [None] * samples * classes
    y = [c for c in ALPHA_NUMERIC for x in xrange(samples)]
    i = 0
    for sample in os.listdir(dirname):
        if not sample.startswith("Sample"):
            continue
        dirpath = os.path.join(dirname, sample)
        for j, imgname in enumerate(os.listdir(dirpath)):
            if j >= samples:
                break
            filepath = os.path.join(dirpath, imgname)
            x = cv2.imread(filepath, 0)
            X[i] = np.array(np.reshape(x, (width**2, ))).astype(np.float32) / 256
            LOGGER.debug("Loaded {} as {}:".format(filepath, y[i]))
            i += 1
    return split_data(X, y, retain, classes)


def load_handwritten(root, width, height, classes=62, thresh=None, retain=0.8):
    """
    Load typed training data.

    Args:
        root (str): Root directory containing training data.
        width (int): Desired width of each sample.
        height (int): Desired height of each sample.
        classes (Optional[int]): Number of classes. Defaults to 62.
        thresh (Optional[int]): Background image threshold. Defaults to None.
        retain (Optional[float]): Percentage of sample data to retain as
            training data. The rest is used as test data. Defaults to 0.8.

    Returns:
        numpy.ndarray: Training data.
        numpy.ndarray: Training labels.
        numpy.ndarray: Test data.
        numpy.ndarray: Test labels.
    """
    train = []
    labels = []
    default_thresh = thresh
    for dirname in os.listdir(root):
        label = dirname[0]
        full_dirname = os.path.join(root, dirname)
        vector = []
        for filename in os.listdir(full_dirname):
            sample = os.path.join(full_dirname, filename)
            img = cv2.imread(sample, 0)


            if default_thresh is None:
                ravel = img.ravel()
                avg = np.mean(ravel)
                thresh = avg
            else:
                thresh = default_thresh
            vec = transform(img, width, height, thresh)
            train.append(vec)
            labels.append(label)

            #show_img(img)
            #img2 = grayscale_to_black_and_white(img, thresh)
            #show_img(img2)
            #img3 = resize(img2, width, height, thresh)
            #show_img(img3)

    return split_data(train, labels, retain, classes)


def load_typed(root, width, height, samples, classes=62, digits=3,
               digits2=5, thresh=128, retain=0.8):
    """
    Load typed training data.

    Args:
        root (str): Root directory containing training data.
        width (int): Desired width of each sample.
        height (int): Desired height of each sample.
        samples (int): Number of samples to use for each class.
        classes (Optional[int]): Number of classes. Defaults to 62.
        digits (Optional[int]): Number of digits in the sample number.
        digits2 (Optional[int]): Number of digits in the image number.
        thresh (Optional[int]): Background image threshold. Defaults to 128.
        retain (Optional[float]): Percentage of sample data to retain as
            training data. The rest is used as test data. Defaults to 0.8.

    Returns:
        numpy.ndarray: Training data.
        numpy.ndarray: Training labels.
        numpy.ndarray: Test data.
        numpy.ndarray: Test labels.
    """
    data = []
    labels = []
    for i in xrange(1, classes + 1):
        dirname = os.path.join(root, "Sample" + str(i).zfill(digits))
        for j in xrange(1, samples + 1):
            filename = os.path.join(
                dirname,
                "img{}-{}.png".format(str(i).zfill(digits), str(j).zfill(digits2)))
            img = cv2.imread(filename, 0)
            vec = transform(img, width, height, thresh)
            data.append(vec)
            labels.append(ALPHA_NUMERIC[i-1])
    return split_data(data, labels, retain, classes)


def load_digits(filename, rows=50, cols=100, width=20, colors=256,
                classes=10, retain=0.8):
    """
    Load training data from file.

    The file contains the mnist digits, but together in one collage that must
    be split apart. This image was obtained from the opencv sample image data.

    Args:
        filename (str): File containing mnist digits.
        rows (Optional[int]): Number of rows in file.
        cols (Optional[int]): Number of cols in file.
        width (Optional[int]): Width and height of each sample.
            Defaults to 20.
        colors (Optional[int]): Number of colors. Defaults to 256.
        classes (Optional[int]): Number of classes. Defaults to 10.
        retain (Optional[float]): Percentage of sample data to retain as
            training data. The rest is used as test data. Defaults to 0.8.

    Returns:
        numpy.ndarray: Training data.
        numpy.ndarray: Training labels.
        numpy.ndarray: Test data.
        numpy.ndarray: Test labels.
    """
    gray = cv2.imread(filename, 0)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, cols) for row in np.vsplit(gray, rows)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Training data
    X = [np.reshape(x[y][x_], (width**2, )).astype(np.float32) / colors
         for x_ in xrange(cols) for y in xrange(rows)]

    # Expected
    y = [y for y in xrange(classes) for x_ in xrange(len(X) / classes)]
    assert len(X) == len(y)

    return split_data(X, y, retain, classes)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("-t", "--training",
                        choices=("handwritten", "typed", "digits",
                                 "shrinked"),
                        required=True,
                        help="Training data type.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--training_dir",
                       help="Directory containing training data. The "
                       "directory structure depends on the training data "
                       "type.")
    group.add_argument("-f", "--training_file",
                       help="File containing training data.")
    parser.add_argument("-k", "--knearest", type=int, default=5,
                        help="Number of neighbors to compare against. "
                        "Defaults to %(default)d.")
    parser.add_argument("--thresh", type=int,
                        help="Background pixel threshold.")
    parser.add_argument("--width", type=int, default=20,
                        help="Sample image width. Defaults to %(default)d.")
    parser.add_argument("--height", type=int, default=20,
                        help="Sample image height. Defaults to %(default)d.")
    parser.add_argument("-r", "--retain", type=float, default=0.8,
                        help="Percentage of sample data to use as training "
                        "data. The rest will be used as test data. This "
                        "number is given as a float from 0.0 to 1.0, not a "
                        "value from 0 to 100. "
                        "Defaults to %(default)0.1f.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples of each class to use. "
                        "Defaults to %(default)d.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--save", nargs="?",
                       const="classifier.{timestamp}.p", default=None,
                       help="filename to save the classifier as. "
                       "Defaults to '%(const)s' if the flag is provided where "
                       "{timestamp} is the current system time.")
    group.add_argument("-p", "--pickle",
                       help="Saved classifier to load and test.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Logging verbosity. More verbose means more "
                        "logging info.")

    args = parser.parse_args()

    # Set logging verbosity
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s",
                        stream=sys.stderr)
    if args.verbose == 1:
        LOGGER.setLevel(logging.INFO)
    elif args.verbose == 2:
        LOGGER.setLevel(logging.DEBUG)

    return args


def main():
    args = get_args()

    if args.training == "handwritten":
        train, train_labels, test, test_labels = load_handwritten(
            args.training_dir, args.width, args.height, thresh=args.thresh,
            retain=args.retain)
    elif args.training == "typed":
        train, train_labels, test, test_labels = load_typed(
            args.training_dir, args.width, args.height, args.samples,
            retain=args.retain)
    elif args.training == "digits":
        train, train_labels, test, test_labels = load_digits(
            args.training_file, retain=args.retain, width=args.width)
    elif args.training == "shrinked":
        train, train_labels, test, test_labels = load_shrinked_imgs(
            args.training_dir, retain=args.retain, width=args.width)
    else:
        raise RuntimeError("Unknown training type: ".format(args.training))

    if args.pickle:
        nbrs = KNN.from_file(args.pickle)
    else:
        clf = NearestNeighbors().fit(train)
        nbrs = KNN(clf, train_labels)
        if args.save:
            nbrs.save(args.save)
    nbrs.test(test, test_labels, k=args.knearest)

    return 0


if __name__ == "__main__":
    sys.exit(main())

