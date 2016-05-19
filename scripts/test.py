#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from random import randint
from ocr.classify import Classifier


def imgfile_to_grayscale(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_digits():
    """Load training data from digits.png"""
    gray = imgfile_to_grayscale("digits.png")

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Training data
    X = [np.reshape(x[y][x_], (400, )).astype(np.float32) / 256
         for x_ in xrange(100) for y in xrange(50)]

    # Expected
    y = [y for y in xrange(10) for x_ in xrange(len(X) / 10)]
    assert len(X) == len(y)

    return X, y


def test_digits(X, y, clf, test=100):
    correct = 0
    for i in xrange(test):
        num = randint(0, 9)
        i = len(X) / 10 * num + randint(0, len(X) / 10 - 1)
        print "expected: ", y[i]

        assert num == y[i]

        prediction = clf.predict([X[i]])[0]

        print "prediction: ", prediction
        if num == prediction:
            correct += 1
    print "% correct: ", correct * 100.0 / test


def main():
    from classify import Classifier

    save_file = "test.p"
    X, y = load_digits()
    x = Classifier()
    x.train(X, y)
    test_digits(X, y, x.mlp)

    x.save(save_file)
    x = Classifier.from_pickle(save_file)
    test_digits(X, y, x.mlp)

    return 0


if __name__ == "__main__":
    main()

