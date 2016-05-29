#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

from knn import (Classifier, img_to_feature, grayscale_to_black_and_white,
                 resize, transform)
from collections import defaultdict
from correct import correct


def showimg(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_regions(ys, height, epsilon=0, min_dist=0, thresh=None):
    """Get the y positions of lines of text."""
    assert len(ys) > 0
    lines = []

    start = None
    #ys = set(ys)
    last = 0
    if thresh is None:
        avg = np.mean(ys)
        std = np.std(ys)
        thresh = max(avg - 2*std, 0)
    counts = defaultdict(int)
    for y in ys:
        counts[y] += 1
    for y in xrange(height):
        if start is None and counts[y] > thresh:
            # Filled line
            start = last
        elif start is not None and counts[y] <= thresh:
            # Empty line
            lines.append([start, y])
            start = None
        last = y

    # Join any tuples that are close together
    #lines_joined = [lines.pop(0)]
    #while lines:
    #    next_line = lines.pop(0)
    #    last_line = lines_joined[-1]
    #    if next_line[0] - last_line[1] <= epsilon:
    #        lines_joined[-1][1] = next_line[1]
    #    else:
    #        lines_joined.append(next_line)
    lines_joined = lines

    # Filter out any regions that are less than min_dist
    lines_joined = [x for x in lines_joined if x[1] - x[0] >= min_dist]

    return lines_joined


def colored_pixels(img, thresh):
    """Get a list of tuples of posiitons of colored pixels in an img."""
    pixels = []
    for y, row in enumerate(img):
        for x, pix in enumerate(row):
            if pix < thresh:
                pixels.append((x, y))
        else:
            continue
        break
    return pixels


def character_regions(img, regions, thresh, epsilon=5, min_dist=0,
                      dark_thresh=0):
    """Split the line regions into characters."""
    lines = []

    w = img.shape[1]
    for start, end in regions:
        sub_img = img[start:end, :]
        assert w == sub_img.shape[1]
        pixels = colored_pixels(sub_img, thresh)
        xs, ys = zip(*pixels)
        char_regions = line_regions(xs, w, epsilon=epsilon, min_dist=min_dist,
                                    thresh=dark_thresh)
        lines.append(char_regions)

    return lines


def imgs_from_regions(img, line_regs, char_regs):
    assert len(char_regs) == len(line_regs)
    for i, char_region in enumerate(char_regs):
        starty, endy = line_regs[i]
        for startx, endx in char_region:
            yield i, img[starty:endy+1, startx:endx+1]


def show_img(img):
    fig = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    fig.show()
    raw_input()
    plt.close(fig)


def img_thresh(img):
    ravel = img.ravel()
    avg = np.mean(ravel)
    std = np.std(ravel)
    thresh = max(0, avg - 2*std)
    return thresh


def identify_chars(img, line_regs, char_regs, classifier, k=5, width=20,
                   height=20, thresh=None, verbose=False):
    """Differentiate between characters and spaces based on the regions."""
    #flat = [char_reg for line_reg in char_regs for char_reg in line_reg]
    #avg_char_dist = sum(x[1] - x[0] for x in flat) * 1.0 / len(flat)
    avg_char_dist = []
    for i in xrange(len(char_regs)):
        last_end = char_regs[i][0][1]
        for j in xrange(1, len(char_regs[i])):
            start, end = char_regs[i][j]
            avg_char_dist.append(start - last_end)
            last_end = end
    avg_char_dist = sum(avg_char_dist) * 1.0 / len(avg_char_dist)

    print("avg char dist:", avg_char_dist)

    chars = []
    count = 0
    for i, char_region in enumerate(char_regs):
        starty, endy = line_regs[i]
        line_chars = []
        last_end = None
        print("Checking line", i)
        for startx, endx in char_region:
            sub_img = img[starty:endy+1, startx:endx+1]

            print("Transforming region", (startx, starty), (endx, endy))
            ravel = sub_img.ravel()
            std = np.std(ravel)
            avg = np.mean(ravel)
            thresh = max(0, avg - 2*std)
            if verbose:
                print("shape:", sub_img.shape)
                print("avg:", avg)
                print("std:", std)
                print("thresh:", thresh)
                x = grayscale_to_black_and_white(sub_img, thresh)
                y = resize(x, width, height, thresh)
                z = grayscale_to_black_and_white(y)
                show_img(sub_img)
                show_img(x)
                show_img(y)
                show_img(z)
                pixels = colored_pixels(z, img_thresh(z))
                xs, ys = zip(*pixels)

                fig = plt.figure()
                plt.imshow(y, cmap="gray")
                plt.scatter(xs, ys, marker=".", color="r")
                fig.show()
                raw_input()
                plt.close(fig)

            sub_img = transform(sub_img, width, height, thresh)

            prediction, guesses, _ = classifier.predictions_and_metadata([sub_img], k=k)
            prediction = prediction[0]
            if verbose:
                print(prediction, guesses)

            # Add space
            if last_end is not None:
                dist = startx - last_end
                if dist > avg_char_dist:
                    chars.append(" ")

            chars.append(prediction)
            count += 1
            last_end = endx
        chars.append("\n")
    return "".join(chars).strip()


def save_images(img, line_regs, char_regs, save_as, dest_dir="characters"):
    assert len(line_regs) == len(save_as) == len(char_regs), "{}, {}, {}".format(len(line_regs), len(save_as), len(char_regs))

    import os
    import errno

    for dirname in save_as:
        try:
            os.makedirs(os.path.join(dest_dir, dirname))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError(e)

    sample = 0
    for i, sub_img in imgs_from_regions(img, line_regs, char_regs):
        filename = os.path.join(dest_dir, save_as[i], str(sample) + ".png")
        cv2.imwrite(filename, sub_img)
        sample += 1
        if sample >= len(char_regs[0]):
            sample = 0


def split_labels(labels):
    return labels.split(",")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("-s", "--save", action="store_true", default=False)
    parser.add_argument("-l", "--labels", type=split_labels, default=[])
    parser.add_argument("--min_line_dist", type=int, default=0)
    parser.add_argument("--min_char_dist", type=int, default=0)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--char_eps", type=int, default=0)
    parser.add_argument("--thresh", type=int)
    parser.add_argument("-p", "--pickle")
    parser.add_argument("-k", "--knearest", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--resize", type=float, default=0.25)

    return parser.parse_args()


def main():
    args = get_args()

    # Get background color
    resize_ratio = args.resize
    img = cv2.imread(args.filename, 0)
    h, w = img.shape[:2]
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    h2, w2 = img.shape[:2]
    print("resized image shape:", img.shape)
    ravel = img.ravel()
    fig1 = plt.figure()
    plt.hist(ravel, 256, [0, 256])
    avg = np.mean(ravel)
    std = np.std(ravel)
    print("std:", std)
    print("mean:", avg)
    print("median:", np.median(ravel))
    print("mean-3std:", avg - 3*std)
    thresh = args.thresh
    if thresh is None:
        thresh = avg - 2*std
    img = grayscale_to_black_and_white(img, 128)
    print("background threshold:", thresh)
    fig1.show()
    print(img)


    # Find top left corner
    text_pixels = colored_pixels(img, thresh)


    fig2 = plt.figure()
    plt.imshow(img, cmap='gray')
    xs, ys = zip(*text_pixels)
    assert len(xs) == len(ys)
    assert max(xs) <= w2
    assert max(ys) <= h2

    plt.scatter(xs, ys, marker=".", color="r")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    fig2.show()

    # Draw line boundaries
    line_positions = line_regions(ys, h2, min_dist=args.min_line_dist,
                                  thresh=0)
    for pos in line_positions:
        plt.plot([0, w2], [pos[0], pos[0]], color="b")
        plt.plot([0, w2], [pos[1], pos[1]], color="b")

    # Draw character boundaries
    char_regions = character_regions(
        img, line_positions, thresh, epsilon=args.char_eps,
        min_dist=args.min_char_dist)
    assert len(char_regions) == len(line_positions)
    for i, char_region in enumerate(char_regions):
        starty, endy = line_positions[i]
        for startx, endx in char_region:
            plt.plot([startx, startx], [starty, endy], color="b")
            plt.plot([endx, endx], [starty, endy], color="b")


    if args.pickle:
        nbrs = Classifier.from_file(args.pickle)
        print("Running classifier", args.pickle)
        text = identify_chars(img, line_positions, char_regions, nbrs,
                              k=args.knearest, thresh=args.thresh,
                              width=args.width, height=args.height,
                              verbose=args.verbose)
        print(text)
        print(" ".join([correct(x) for x in text.split()]))
    elif args.save:
        save_images(img, line_positions, char_regions, args.labels)
    else:
        raw_input()  # Keep figures alive
    return 0


if __name__ == "__main__":
    main()

