# -*- coding; utf-8 -*-

from __future__ import print_function

import numpy as np
import os

from ocr.utils import imgfile_to_grayscale
from ocr.classify import Classifier


def load_digits(digits_filename):
    """Load the mnist data set from the given digits file."""
    gray = imgfile_to_grayscale(digits_filename)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    arr = np.array(cells)

    # Training data
    X = [np.reshape(arr[y][x], (400,)).astype(np.float32) / 256
         for x in xrange(100) for y in xrange(50)]

    # Expected
    y = [y for y in xrange(10) for x_ in xrange(len(X) / 10)]
    assert len(X) == len(y)

    return X, y


def split_digits(X, y, percentage):
    """Split the data set into training and test data."""
    test_X = []
    test_y = []
    train_X = []
    train_y = []
    digits = 10;
    for i in xrange(digits):
        start_idx = len(y) / digits * i
        split_idx = int(start_idx + len(y) / digits * percentage)
        end_idx = len(y) / digits * (i + 1)
        train_X += X[start_idx:split_idx]
        train_y += y[start_idx:split_idx]
        test_X += X[split_idx:end_idx]
        test_y += y[split_idx:end_idx]
    return train_X, train_y, test_X, test_y


def test_digits(X, y, clf):
    from random import randint

    assert len(X) == len(y)
    correct = 0
    for i in xrange(len(y)):
        print("expected: ", y[i])
        prediction = clf.predict([X[i]])[0]
        print("prediction: ", prediction)
        if y[i] == prediction:
            correct += 1
    print("% correct: ", correct * 100.0 / len(X))


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser("Load MNIST training set")

    parser.add_argument("filename", help="File containing mnist data.")
    parser.add_argument("-p", "--pickle", help="Saved classifier to reuse.")
    parser.add_argument("-n", "--nodes", type=int, default=100,
                        help="Number of nodes in each hidden layer.")
    parser.add_argument("-l", "--layers", type=int, default=25,
                        help="Number of hidden layers.")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum number of iterations when "
                        "backpropagating.")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Initial learning rate.")
    parser.add_argument("-s", "--save", help="File to save classifier as.")
    parser.add_argument("-r", "--random_state", default=None, type=int,
                        help="Random seed to use with classifier.")
    parser.add_argument("--training_percentage", type=int, default=80,
                        help="Percentage of digits to use as training data. "
                        "The rest is used as test data.")

    return parser.parse_args()


def main():
    args = get_args()

    X, y = load_digits(args.filename)

    # Split into training and test
    train_X, train_y, test_X, test_y = split_digits(
        X, y, args.training_percentage / 100.0)

    if args.pickle:
        clf = Classifier.from_pickle(args.pickle)
    else:
        clf = Classifier(hidden_layer_sizes=(args.nodes, args.layers),
                         verbose=True,
                         activation="tanh",
                         max_iter=args.max_iter,
                         algorithm="sgd",
                         learning_rate_init=args.learning_rate,
                         random_state=args.random_state)
        clf.fit(train_X, train_y)
        if args.save:
            clf.save(args.save)
        else:
            clf.save()

    test_digits(test_X, test_y, clf)
    #test_digits(train_X, train_y, clf)

    return 0


if __name__ == "__main__":
    main()

