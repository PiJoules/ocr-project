#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
import cv2
import unittest

from random import randint
from ocr.classify import Classifier
from tempfile import NamedTemporaryFile


PARENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(PARENT_DIR), "data")


class TestDigits(unittest.TestCase):
    """Test case for classifying handwritten digits."""

    TRHESH = 0.95
    SAVE_FILE = "TestDigitsClassifier.p"
    DIGITS_FILE = os.path.join(DATA_DIR, "digits.png")

    def test_classify(self):
        # Load digits
        X, y = self.__load_digits()

        # Train classifier
        seed = 1
        clf = Classifier(random_state=seed)
        correct = clf.train(X, y)

        # Test
        self.__test_digits(X, y, clf)

        # Test save
        clf.save(self.SAVE_FILE)
        clf = Classifier.from_pickle(self.SAVE_FILE)
        self.assertEqual(correct, self.__test_digits(X, y, clf))

    def __test_digits(self, X, y, clf):
        """Test that the digits are classified correctly by a classifier."""
        self.assertEqual(len(X), len(y))
        correct = 0
        for i in xrange(len(y)):
            expected = y[i]
            prediction = clf.classify([X[i]])[0]
            if expected == prediction:
                correct += 1

        self.assertGreaterEqual(correct, self.TRHESH * len(y))
        return correct

    def __imgfile_to_grayscale(self, filename):
        """Load an image as grayscale."""
        img = cv2.imread(filename)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __load_digits(self):
        """Load training data from digits.png"""
        gray = self.__imgfile_to_grayscale(self.DIGITS_FILE)

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


if __name__ == "__main__":
    unittest.main()

