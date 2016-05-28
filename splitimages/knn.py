#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from cropimage import trimmed_image, pad_and_resize
from main import colored_pixels


def grayscale_to_black_and_white(img, thresh):
    return np.array([[0 if pix <= thresh else 255 for pix in row] for row in img]).astype(np.uint8)


def img_to_feature(img):
    return np.reshape(img, (img.shape[0] * img.shape[1], ))


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("trainingdir")
    parser.add_argument("-k", "--knearest", type=int, default=5)
    parser.add_argument("--thresh", type=int)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)

    return parser.parse_args()


def main():
    args = get_args()

    train = []
    labels = []
    thresh = args.thresh
    for dirname in os.listdir(args.trainingdir):
        label = dirname[0]
        full_dirname = os.path.join(args.trainingdir, dirname)
        vector = []
        for filename in os.listdir(full_dirname):
            sample = os.path.join(full_dirname, filename)
            img = cv2.imread(sample, 0)
            ravel = img.ravel()
            avg = np.mean(ravel)
            std = np.std(ravel)
            if thresh is None:
                thresh = avg - 2*std
            #print(sample)
            #print("img size:", img.shape)
            #print("color avg:", avg)
            #dark = colored_pixels(img, thresh)
            #img = grayscale_to_black_and_white(img, thresh)
            trimmed = trimmed_image(img, thresh=thresh)
            resized = pad_and_resize(trimmed, args.width, args.height,
                                     bg=255)
            resized = grayscale_to_black_and_white(resized, thresh)
            #vector.append(img_to_feature(resized))
            train.append(img_to_feature(resized))
            labels.append(label)
            #dark = colored_pixels(resized, thresh)
            #xs, ys = zip(*dark)

            #fig = plt.figure()

            ##plt.imshow(img, cmap="gray")
            #plt.imshow(trimmed, cmap="gray")
            ##plt.imshow(resized, cmap="gray")
            ##plt.scatter(xs, ys, marker=".", color="r")
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #fig.show()

            #fig2 = plt.figure()
            #plt.imshow(resized, cmap="gray")
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #fig2.show()

            ##fig3 = plt.figure()
            ##bw = grayscale_to_black_and_white(img, thresh)
            ##plt.imshow(bw, cmap="gray")
            ##fig3.show()

            ### Pause to keep img alive
            #raw_input()
            #plt.close(fig)
            #plt.close(fig2)
            ##plt.close(fig3)
            #print("loaded:", sample)
        #train.append(vector)
        #labels.append(label)
    train = np.array(train)
    labels = np.array(labels)

    nbrs = NearestNeighbors(n_neighbors=args.knearest).fit(train, y=labels)
    distances, indeces = nbrs.kneighbors([train[0]])
    print(distances)
    print(indeces)

    return 0


if __name__ == "__main__":
    main()

