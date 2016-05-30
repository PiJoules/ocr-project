# -*- coding: utf-8 -*-

from __future__ import print_function

import pickle
import logging

from datetime import datetime

LOGGER = logging.getLogger(__name__)


class Classifier(object):

    def __init__(self, clf, labels, **kwargs):
        self.__clf = clf
        self.__labels = labels

    @classmethod
    def from_file(cls, filename):
        """Load this classifier from a saved pickle file."""
        with open(filename, "rb") as p:
            return pickle.load(p)

    def classifier(self):
        return self.__clf

    def training_labels(self):
        return self.__labels

    def predictions_and_metadata(self, data, **kwargs):
        """
        Make a prediction from data and return any metadata associated with
        the prediction.

        Args:
            data (numpy.ndarray): Data to classify.
            **kwargs: Keyword arguments passed to the classifier's prediction
                method.

        Returns:
            numpy.ndarray: Predictions
            any: Other metadata associated with the predictions
        """
        raise NotImplementedError

    def predict(self, data, **kwargs):
        """Just make a prediction from an array of feature vectors."""
        return self.predictions_and_metadata(data, **kwargs)[0]

    def save(self, filename="classifier.{timestamp}.p"):
        """Save this classifier for later use."""
        filename = filename.format(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(filename, "wb") as save:
            pickle.dump(self, save)

    def test(self, data, expected, **kwargs):
        """Test the classifier with test data."""
        predictions = self.predict(data, **kwargs)
        correct = 0
        for i, x in enumerate(expected):
            pred = predictions[i]
            LOGGER.debug("expected: {}, guess: {}".format(x, pred))
            if x == pred:
                correct += 1
        print(correct * 100.0 / len(expected), "%")

