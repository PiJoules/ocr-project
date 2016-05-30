#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import pickle
import logging
import cv2
import string
import os
import numpy as np

from sklearn.neural_network import MLPClassifier
from ocr.utils import split_data, default_background_threshold
from ocr.classifiers.base import Classifier

LOGGER = logging.getLogger(__name__)
ALPHA_NUMERIC = string.digits + string.ascii_uppercase + string.ascii_lowercase


class MLP(Classifier):
    """Character recognizer"""

    def predictions_and_metadata(self, data, **kwargs):
        return tuple([self.classifier().predict(data, **kwargs)])


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


def load_english_hand(base_dir, samples=55, width=20, retain=0.8,
                      classes=62, thresh=None, colors=256):
    """
    Load english handwritten characters  (using pc tablet) from
    http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

    Samples (55 of each; total 3410 samples)
    0-10:
        Numbers 0-9
    11-36:
        Uppercase
    37-62:
        Lowercase

    Args:
        base_dir (str): Directory containing sample data.
        samples (Optional[int]): Number of samples for each class.
        width (Optional[int]): Width and height of each sample.
            Defaults to 20.
        retain (Optional[float]): Percentage of sample data to retain as
            training data. The rest is used as test data. Defaults to 0.8.
        classes (Optional[int]): Number of classes. Defaults to 62.
        thresh (Optional[int]): Background threshold.
        colors (Optional[int]): Number of colors. Defaults to 256.

    Returns:
        numpy.ndarray: Training data.
        numpy.ndarray: Training labels.
        numpy.ndarray: Test data.
        numpy.ndarray: Test labels.
    """
    import time

    start = time.time()
    X = [None] * samples * classes
    y = [c for c in ALPHA_NUMERIC for x in xrange(samples)]
    default_thresh = thresh
    with open(os.path.join(base_dir, "all.txt~"), "r") as samples:
        for i, sample in enumerate(samples):
            sample = sample.strip()
            filename = os.path.join(base_dir, sample)
            img = cv2.imread(filename, 0)
            if default_thresh is None:
                thresh = default_background_threshold(img)
            else:
                thresh = default_thresh

            vec = transform(img, width, width, thresh)

            X[i] = vec.astype(np.float32) / colors
            LOGGER.debug("Loaded: {}".format(i))
    LOGGER.info("Loading training set: {} seconds".format(time.time() - start))

    return split_data(X, y, retain, classes)


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


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("-t", "--training", required=True,
                        choices=("digits", "english", "shrinked"),
                        help="Training data type.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--training_dir",
                       help="Directory containing training data. The "
                       "directory structure depends on the training data "
                       "type.")
    group.add_argument("-f", "--training_file",
                       help="File containing training data.")
    parser.add_argument("--thresh", type=int,
                        help="Background pixel threshold.")
    parser.add_argument("--width", type=int, default=20,
                        help="Sample image width. Defaults to %(default)d.")
    parser.add_argument("-r", "--retain", type=float, default=0.8,
                        help="Percentage of sample data to use as training "
                        "data. The rest will be used as test data. This "
                        "number is given as a float from 0.0 to 1.0, not a "
                        "value from 0 to 100. "
                        "Defaults to %(default)0.1f.")
    parser.add_argument("--samples", type=int, default=55,
                        help="Number of samples of each class to use. "
                        "Defaults to %(default)d.")
    parser.add_argument("-n", "--nodes", type=int, default=100,
                        help="Number of nodes in each hidden layer. "
                        "Defaults to %(default)d.")
    parser.add_argument("-l", "--layers", type=int, default=25,
                        help="Number of hidden layers. "
                        "Defaults to %(default)d.")
    parser.add_argument("--activation", default="tanh",
                        help="Activation function for the hidden layer.")
    parser.add_argument("--algorithm", default="sgd",
                        help="The algorithm for weight optimization.")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum number of iterations.")
    parser.add_argument("--learning_rate_init", type=float, default=0.01,
                        help="The initial learning rate used.")
    parser.add_argument("--random_state", type=int, default=None,
                        help="State or seed for random number generator.")

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

    # Get training data
    if args.training == "digits":
        train, train_labels, test, test_labels = load_digits(
            args.training_file, retain=args.retain, width=args.width)
    elif args.training == "english":
        train, train_labels, test, test_labels = load_english_hand(
            args.training_dir, retain=args.retain, width=args.width,
            thresh=args.thresh, samples=args.samples)
    elif args.training == "shrinked":
        train, train_labels, test, test_labels = load_shrinked_imgs(
            args.training_dir, retain=args.retain, width=args.width,
            samples=args.samples)
    else:
        raise RuntimeError("Unknown training type: ".format(args.training))

    # Test
    if args.pickle:
        nbrs = KNN.from_file(args.pickle)
    else:
        # Create classifier
        hls = (args.nodes, args.layers)
        clf = MLPClassifier(hidden_layer_sizes=hls, verbose=bool(args.verbose),
                            activation=args.activation, max_iter=args.max_iter,
                            algorithm=args.algorithm,
                            learning_rate_init=args.learning_rate_init,
                            random_state=args.random_state)
        clf.fit(train, train_labels)
        mlp = MLP(clf, train_labels)

        if args.save:
            mlp.save(args.save)
    mlp.test(test, test_labels)

    return 0


if __name__ == "__main__":
    sys.exit(main())

