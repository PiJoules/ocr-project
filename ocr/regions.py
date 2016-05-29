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


def background_threshold(img):
    """
    Get the background threshold for a grayscale image as the average pixel
    color minus 2*the std of the pixel color.

    If this value is negative, 0 is returned instead.
    """
    assert len(img.shape) == 2
    ravel = img.ravel()
    avg = np.mean(ravel)
    std = np.std(ravel)
    bg_thresh = max(0, avg - 2*std)
    return bg_thresh


def colored_pixels(img, thresh):
    """
    Find the positions of all pixels in the image that are considered dark
    enough for text.

    The image is assumed to be grayscale.

    Args:
        img (numpy.ndarray): Image to check.
        thresh (int): Background threshold up to which a pixel is considered
            colored.

    Returns:
        list[tuple[int, int]]: List of coordinates for all colored pixels.
    """
    assert len(img.shape) == 2  # 2D matrix
    return [(x, y) for y, row in enumerate(img) for x, pixel in enumerate(row)
            if pixel <= thresh]


def text_regions(positions, length, min_dist=0, min_pixels=1):
    """
    Find the points of an image along an axis that separate regions containing
    text. This is used to find lines of text in an image and characters in each
    of these lines.

    This function assumes the image is just composed of dark text on a light
    background (grayscaled). The image is also assumed to not be rotated.

    The boundaries for the regions start and end on the whitespace around the
    text, so the pixels for the actual text are in between the regions.

    Args:
        positions (list[int]): Positions along axis of all pixels that should
            be separated into regions.
            For example, when splitting an image with three lines of text in it,
            and the image is composed of black text on a white background,
            positions would be the y-coordinates of all black pixels on
            the image. A majority of these y-coords will be clustered
            into three different regions along the y-axis, so this function
            will return three regions.
        length (int): Length of the image along an axis.
            For example, when splitting an image into lines of text, length
            will be the height of the image.
        min_dist (Optional[int]): Minimum distance for a region. Regions with
            a length less than the min_dist will not be returned. Defaults
            to 0.
        min_pixels (Optional[int]): The minimum number of pixels required for
            a position along the axis to contain text. For example, a position
            along the axis that contains less than this value will be
            considered empty since it does not have enough dark pixels to have
            text. Defaults to 1.
            This value cannot be less than 1 since that would mean all
            positions along the axis are regions of lines.

    Returns:
        list[tuple[int, int]]: List of regions where text is contained in the
            image. These tuples are the boundaries of the text along an axis
            with the pixels of the text between these boundaries.
    """
    assert len(positions) > 0, "No distribution provided."
    assert min_pixels >= 1, ("min_pixels must be at least 1, otherwise all "
        "positions along an axis are considered as having text")

    regions = []  # Regions containing text
    start = None  # Starting boundary for a region
    last_pos = 0

    # Dict tracking the number of times a dark pixel is at a given position
    counts = defaultdict(int)
    for pos in positions:
        counts[pos] += 1

    for pos in xrange(length):
        if start is None and counts[pos] >= min_pixels:
            # Start of region
            start = last_pos
        elif start is not None and counts[pos] < min_pixels:
            # End of region
            regions.append((start, pos))
            start = None
        last_pos = pos

    # Filter out any regions whose length is less than min_dist
    regions = [x for x in regions if x[1] - x[0] >= min_dist]

    return regions


def line_regions(img, bg_thresh=None, **kwargs):
    """
    Find the regions of a grayscale image that contain lines if text.

    Args:
        img (numpy.ndarray): Grayscaled image with dark text on a light
            background.
        bg_thresh (Optional[int]): Background threshold up to which a pixel on
            the image is considered text and not part of the background color.
            If this is not provided (bg_thresh is None), the threshold will be
            the average pixel color of the image minus 2*the std of pixel
            colors, or 0 if this value is negative.
        **kwargs: Keyword arguments passed to text_regions.

    Returns:
        list[tuple[int, int]]: List of regions along the y-axis where lines
            of text in the image start and end.
    """
    if bg_thresh is None:
        bg_thresh = background_threshold(img)

    assert bg_thresh >= 0, "bg_thresh cannot be negative."
    assert len(img.shape) == 2

    x_distr, y_distr = zip(*colored_pixels(img, bg_thresh))

    return text_regions(y_distr, img.shape[0], **kwargs)


def character_regions(img, line_regs, bg_thresh=None, **kwargs):
    """
    Find the characters in an image given the regions of lines if text in the
    image.

    Args:
        img (numpy.ndarray): Grayscaled image.
        line_regs (list[tuple[int, int]]): List of regions representing where
            the lines are.
        bg_thresh (Optional[int]): Background threshold up to which a pixel
            is considered text and not part of the background. If not provided,
            a default background threshold is calculated for each line region
            in the image and used instead.
        **kwargs: Keyword arguments passed to text_regions.
    """
    assert len(img.shape) == 2

    regions = []
    w = img.shape[1]

    for start, end in line_regs:
        sub_img = img[start:end+1, :]

        if bg_thresh is None:
            bg_thresh = background_threshold(sub_img)

        # Sanity check
        assert w == sub_img.shape[1]

        pixels = colored_pixels(sub_img, bg_thresh)
        x_distr, y_distr = zip(*pixels)
        char_regions = text_regions(x_distr, w, **kwargs)
        regions.append(char_regions)

    return regions


def imgs_from_regions(img, line_regs, char_regs):
    """
    Iterate through the characters in an image given the line regions and
    character regions.

    Yields:
        int: yth line
        int: xth character in the yth line
        numpy.ndarray: sub image in the region
    """
    assert len(char_regs) == len(line_regs)
    for y, char_region in enumerate(char_regs):
        starty, endy = line_regs[y]
        for x, (startx, endx) in char_region:
            yield y, x, img[starty:endy+1, startx:endx+1]


def show_img(img, cmap="gray"):
    """Just display an image."""
    fig = plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.xticks([]), plt.yticks([])  # Hide tick values on X and Y axis
    fig.show()

    # Keep image alive by waiting for user input
    raw_input("Press 'Enter' to close the image.")
    plt.close(fig)


def avg_dist_between_chars(char_regs):
    """Get the average distance between each char from the char reigons."""
    count = total = 0
    for i in xrange(len(char_regs)):
        last_end = char_regs[i][0][1]
        for j in xrange(1, len(char_regs[i])):
            start, end = char_regs[i][j]
            total += start - last_end
            count += 1
            last_end = end
    return total * 1.0 / count


def get_text(img, line_regs, char_regs, classifier, k=5, width=20,
             height=20, thresh=None, verbose=False):
    """
    Try to extract text from an image given the line regions and character
    regions.
    """
    avg_char_dist = avg_dist_between_chars(char_regs)
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
                pixels = colored_pixels(z, background_threshold(z))
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
    for i, _, sub_img in imgs_from_regions(img, line_regs, char_regs):
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
    line_positions = text_regions(ys, h2, min_dist=args.min_line_dist,
                                  thresh=1)
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
        text = get_text(img, line_positions, char_regions, nbrs,
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

