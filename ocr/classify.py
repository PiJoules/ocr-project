# -*- coding: utf-8 -*-

from __future__ import print_function

import pickle

from sklearn.neural_network import MLPClassifier


class Classifier(object):
    """Character recognizer"""

    TS_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Keyword arguments fed to the MLPClassifier.
                These are some important ones that were optimal when training
                for just digits.

                hidden_layer_sizes (Any): Either a tuple of length 2 where the
                    first element is the number of nodes in each layer and the
                    second is the number of layers, or a single int
                    representing the number of nodes for 1 layer.
                activation (str): Activation function. Defaults to "tanh".
                max_iter (int): Maximum number of iterations when training
                    with backpropagation. Defaults to 1000.
                algorithm (str): The algrotihm for optomizing the loss function.
                    Defaults to "sgd" for stochastic gradient descent.
                learning_rate_init (float): Initial learning rate. Defaults to
                    0.01.
        """
        hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100, 25))
        activation = kwargs.get("activation", "tanh")
        max_iter = kwargs.get("max_iter", 1000)
        algorithm = kwargs.get("algorithm", "sgd")
        learning_rate_init = kwargs.get("learning_rate_init", 0.01)

        self.__clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            max_iter=max_iter, algorithm=algorithm,
            learning_rate_init=learning_rate_init, **kwargs)

    @classmethod
    def from_pickle(cls, filename):
        """Load the class from a pickle file."""
        with open(filename, "rb") as save_file:
            return pickle.load(save_file)

    @property
    def mlp(self):
        return self.__clf

    def train(self, data, labels):
        """
        Train the classifier. This must be called before classify, otherwise
        the classifier will throw an error since it cannot classify if it was
        not trained.

        Args:
            data (list): List of 1D feature vectors.
            labels (list): 1D list of labels for each corresponding feature
                vector in data.
        """
        self.__clf.fit(data, labels)

    def classify(self, imgs):
        """
        Classify images.

        Args:
            imgs (list): List of images represented as feature vectors to
                classify.

        Returns:
            List of labels the classifier has predicted the given imgs to be.
        """
        return self.__clf.predict(imgs)

    def save(self, filename="classifier.{timestamp}.p"):
        """
        Save this instance of the classifier in a pickle file.
        Will substitute in the current time if '{timestamp}' is provided in
        the filename.
        """
        from datetime import datetime

        timestamp = datetime.now().strftime(self.TS_FORMAT)
        filename = filename.format(timestamp=timestamp)
        with open(filename, "wb") as save_file:
            pickle.dump(self, save_file)

