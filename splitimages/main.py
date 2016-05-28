#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np


def showimg(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_regions(ys, height, epsilon=0, min_dist=0):
    """Get the y positions of lines of text."""
    lines = []

    start = None
    ys = set(ys)
    last = 0
    for y in xrange(height):
        if start is None and y in ys:
            # Found start of line
            start = last
        elif start is not None and y not in ys:
            lines.append([start, y])
            start = None
        last = y

    # Join any tuples that are close together
    lines_joined = [lines.pop(0)]
    while lines:
        next_line = lines.pop(0)
        last_line = lines_joined[-1]
        if next_line[0] - last_line[1] <= epsilon:
            lines_joined[-1][1] = next_line[1]
        else:
            lines_joined.append(next_line)

    # Filter out any regions that are less than min_dist
    lines_joined = [x for x in lines_joined if x[1] - x[0] >= min_dist]

    return lines_joined


def pixel_is_near_colored(img, px, py, thresh, dist=10):
    h, w = img.shape[:2]
    for y in xrange(max(0, py - dist), min(h, py + dist + 1)):
        for x in xrange(max(0, px - dist), min(w, px + dist + 1)):
            pix = img[y, x]
            if pix < thresh:
                return True
    return False


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


def character_regions(img, regions, thresh, epsilon=5, min_dist=10):
    """Split the line regions into characters."""
    lines = []

    w = img.shape[1]
    for start, end in regions:
        sub_img = img[start:end, :]
        assert w == sub_img.shape[1]
        pixels = colored_pixels(sub_img, thresh)
        xs, ys = zip(*pixels)
        char_regions = line_regions(xs, w, epsilon=epsilon, min_dist=min_dist)
        lines.append(char_regions)

    return lines


def imgs_from_regions(img, line_regs, char_regs):
    assert len(char_regs) == len(line_regs)
    for i, char_region in enumerate(char_regs):
        starty, endy = line_regs[i]
        for startx, endx in char_region:
            yield i, img[starty:endy, startx:endx]


def save_images(img, line_regs, char_regs, save_as, dest_dir="characters"):
    assert len(line_regs) == len(save_as) == len(char_regs), "{}, {}, {}".format(len(line_regs), len(save_as), len(char_regs))
    #for i in xrange(1, len(char_regs)):
    #    assert len(char_regs[i]) == len(char_regs[0]), "char_regs[{}]: {}, expected: {}".format(i, len(char_regs[i]), len(char_regs[0]))

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
    parser.add_argument("--min_line_dist", type=int, default=10)
    parser.add_argument("--min_char_dist", type=int, default=10)
    parser.add_argument("--char_eps", type=int, default=0)
    parser.add_argument("--thresh", type=int)

    return parser.parse_args()


def main():
    args = get_args()

    # Get background color
    resize_ratio = 0.25
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
    print("background threshgold:", thresh)
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

    # Draw line boundaries
    line_positions = line_regions(ys, h2, min_dist=args.min_line_dist)
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

    plt.scatter(xs, ys, marker=".", color="r")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    fig2.show()

    if args.save:
        save_images(img, line_positions, char_regions, args.labels)
    else:
        raw_input()  # Keep figures alive
    return 0


if __name__ == "__main__":
    main()

