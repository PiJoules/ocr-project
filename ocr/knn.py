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
import matplotlib.pyplot as plt
import pickle
import string
import logging

from sklearn.neighbors import NearestNeighbors
from cropimage import trimmed_image, pad_and_resize
from datetime import datetime

LOGGER = logging.getLogger(__name__)
ALPHA_NUMERIC = string.digits + string.ascii_uppercase + string.ascii_lowercase


def grayscale_to_black_and_white(img, thresh=None):
    if thresh is None:
        ravel = img.ravel()
        avg = np.mean(ravel)
        std = np.std(ravel)
        thresh = max(0, avg - 2*std)
    return np.array([[0 if pix <= thresh else 255 for pix in row] for row in img]).astype(np.uint8)


def resize(img, width, height, thresh):
    trimmed = trimmed_image(img, thresh=thresh)
    img = pad_and_resize(trimmed, width, height, bg=255)
    return img


def img_to_feature(img):
    return np.reshape(img, (img.shape[0] * img.shape[1], ))


def transform(img, width, height, thresh):
    img = grayscale_to_black_and_white(img, thresh)
    img = resize(img, width, height, thresh)
    img = img_to_feature(img)
    return img


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


def dict_from_data(data, labels):
    return {d: labels[i] for i, d in enumerate(data)}


class Classifier(object):

    def __init__(self, train, labels, **kwargs):
        self.__clf = NearestNeighbors(**kwargs).fit(train)
        self.__labels = labels

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as p:
            return pickle.load(p)

    def predictions_and_metadata(self, data, k=5):
        distances, indeces = self.__clf.kneighbors(data, n_neighbors=k)
        label_pred = [[self.__labels[i] for i in idxs] for idxs in indeces]
        predictions = [max(idxs, key=idxs.count) for idxs in label_pred]
        return predictions, label_pred, distances

    def predict(self, data, **kwargs):
        return self.predictions_and_metadata(data, **kwargs)[0]

    def save(self, filename="classifier.{timestamp}.p"):
        filename = filename.format(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(filename, "wb") as save:
            pickle.dump(self, save)

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


def load_handwritten(args):
    train = []
    labels = []
    for dirname in os.listdir(args.training_dir):
        label = dirname[0]
        full_dirname = os.path.join(args.training_dir, dirname)
        vector = []
        for filename in os.listdir(full_dirname):
            sample = os.path.join(full_dirname, filename)
            img = cv2.imread(sample, 0)
            if args.thresh is None:
                ravel = img.ravel()
                avg = np.mean(ravel)
                thresh = avg
            else:
                thresh = args.thresh
            grayscale = grayscale_to_black_and_white(img, thresh)
            resized = resize(grayscale, args.width, args.height, thresh)
            feature = img_to_feature(resized)
            train.append(feature)
            labels.append(label)

    return split_data(train, labels, args.retain, 62)


def load_typed(root, width, height, samples, classes=62, digits=3,
               digits2=5, thresh=128, retain=0.8):
    """
    Load typed training data.
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
            resized = resize(img, width, height, thresh)
            feature = img_to_feature(resized)
            data.append(feature)
            labels.append(ALPHA_NUMERIC[i-1])
    return split_data(data, labels, retain, classes)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("-t", "--training", choices=("handwritten", "typed"),
                        required=True,
                        help="Training data type.")
    parser.add_argument("-d", "--training_dir", required=True,
                        help="Directory containing training data. The "
                        "directory structure depends on the training data "
                        "type.")
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
        train, train_labels, test, test_labels = load_handwritten(args)
    elif args.training == "typed":
        train, train_labels, test, test_labels = load_typed(
            args.training_dir, args.width, args.height, args.samples,
            retain=args.retain)
    else:
        raise RuntimeError("Unknown training type: ".format(args.training))

    if args.pickle:
        nbrs = Classifier.from_file(args.pickle)
    else:
        nbrs = Classifier(train, train_labels)
        if args.save:
            nbrs.save(args.save)
    nbrs.test(test, test_labels, k=args.knearest)

    return 0


if __name__ == "__main__":
    main()

