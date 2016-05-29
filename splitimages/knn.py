#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.neighbors import NearestNeighbors
from cropimage import trimmed_image, pad_and_resize
from datetime import datetime


def grayscale_to_black_and_white(img, thresh):
    return np.array([[0 if pix <= thresh else 255 for pix in row] for row in img]).astype(np.uint8)


def resize(img, width, height, thresh):
    trimmed = trimmed_image(img, thresh=thresh)
    img = pad_and_resize(trimmed, width, height, bg=255)
    return img


def img_to_feature(img):
    #img = resize(img, width, height, thresh)
    #img = grayscale_to_black_and_white(img, thresh)
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


def load_data(args):
    train = []
    labels = []
    for dirname in os.listdir(args.trainingdir):
        label = dirname[0]
        full_dirname = os.path.join(args.trainingdir, dirname)
        vector = []
        for filename in os.listdir(full_dirname):
            sample = os.path.join(full_dirname, filename)
            img = cv2.imread(sample, 0)
            if args.thresh is None:
                ravel = img.ravel()
                avg = np.mean(ravel)
                #std = np.std(ravel)
                #thresh = avg - 2*std
                thresh = avg
            else:
                thresh = args.thresh
            grayscale = grayscale_to_black_and_white(img, thresh)
            resized = resize(grayscale, args.width, args.height, thresh)
            #feature = img_to_feature(img, args.width, args.height, thresh)
            feature = img_to_feature(resized)
            train.append(feature)
            labels.append(label)

    return split_data(train, labels, args.retain, 62)


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
            print("expected:", x, ", guess:", pred, ", guesses:", all_predictions[i])
            if x == pred:
                correct += 1
        print(correct * 100.0 / len(expected))


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("trainingdir")
    parser.add_argument("-k", "--knearest", type=int, default=5)
    parser.add_argument("--thresh", type=int)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("-r", "--retain", type=float, default=0.8)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--save", nargs="?",
                       const="classifier.{timestamp}.p", default=None)
    group.add_argument("-p", "--pickle")

    return parser.parse_args()


def main():
    args = get_args()

    train, train_labels, test, test_labels = load_data(args)
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

