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
SEED = 1


class TestDigits(unittest.TestCase):
    """Test case for classifying handwritten digits."""

    TRHESH = 0.85
    SAVE_FILE = "TestDigitsClassifier.p"
    DIGITS_FILE = os.path.join(DATA_DIR, "digits.png")

    @classmethod
    def setUpClass(cls):
        # Load digits
        cls.__X, cls.__y = cls.load_digits()

        # Train classifier
        clf = Classifier(random_state=SEED)
        clf.train(cls.__X, cls.__y)
        cls.__clf = clf

    def test_classify(self):
        """Test results"""
        self.__test_digits(self.__X, self.__y, self.__clf)

    def test_save(self):
        """Test save"""
        results = self.__test_digits(self.__X, self.__y, self.__clf)
        clf = self.__clf
        clf.save(self.SAVE_FILE)
        clf = Classifier.from_pickle(self.SAVE_FILE)
        self.assertEqual(results, self.__test_digits(self.__X, self.__y, clf))

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

    @staticmethod
    def imgfile_to_grayscale(filename):
        """Load an image as grayscale."""
        img = cv2.imread(filename)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @classmethod
    def load_digits(cls):
        """Load training data from digits.png"""
        gray = cls.imgfile_to_grayscale(cls.DIGITS_FILE)

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

