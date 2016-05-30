#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

from ocr.utils import (default_background_threshold,
                       grayscale_to_black_and_white, transform)
from collections import defaultdict
from correct import correct

LOGGER = logging.getLogger(__name__)


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
        bg_thresh = default_background_threshold(img)

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
            bg_thresh = default_background_threshold(sub_img)

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


def get_text_from_regions(img, line_regs, char_regs, classifier, bg_thresh=None,
                          resize=None, **kwargs):
    """
    Try to extract text from an image given the line and character
    regions, and classifier.

    Args:
        img (numpy.ndarray): Image to extract text from. This image does not
            need to be grayscaled since it will be done in this function, but
            the image should be unrotated and be a dark text on a light
            background.
        line_regs (list[tuple[int, int]]): Line regions in image.
        char_regs (list[list[tuple[int, int]]]): Character regions in each
            line in the image.
        classifier (Classifier): Classifier to use to predict characters.
        resize (Optional[tuple[int, int]): Dimensions to resize the image to.
            Defaults to None (indicating we should not resize). If provided,
            it should be a tuple containing the width and height.
            Ex: (width, height)
        **kwargs: Keyword arguments to pass to classifier predict method.

    Returns:
        str: The string extracted from the image.
    """
    avg_char_dist = avg_dist_between_chars(char_regs)
    LOGGER.debug("avg char dist: {}".format(avg_char_dist))

    chars = ""
    default_thresh = bg_thresh
    for i, char_region in enumerate(char_regs):
        starty, endy = line_regs[i]
        last_end = None
        LOGGER.debug("Checking line {}".format(i))
        for startx, endx in char_region:
            sub_img = img[starty:endy+1, startx:endx+1]

            LOGGER.debug("Transforming region between points {} {}"
                         .format((startx, starty), (endx, endy)))

            if default_thresh is None:
                bg_thresh = default_background_threshold(sub_img)
            LOGGER.debug("background threshold: {}".format(bg_thresh))

            if resize is not None:
                sub_img = transform(sub_img, resize[0], resize[1], bg_thresh)
            prediction = classifier.predict([sub_img], **kwargs)[0]
            LOGGER.debug("prediction: {}".format(prediction))

            # Add space if characters are far enough apart
            if last_end is not None:
                dist = startx - last_end
                if dist > avg_char_dist:
                    chars += " "

            chars += prediction
            last_end = endx
        chars += "\n"
    return chars


def get_text(img, classifier, bg_thresh=None, resize=None, min_char_dist=0,
             min_char_pixels=1, min_line_dist=0, min_line_pixels=1,
             spell_check=False, **kwargs):
    """
    Get the text of an image with a classifier.

    Args:
        get_text_from_regions() arguments:
            img
            classifier
            bg_tresh
            resize
            **kwargs

        min_line_dist (Optional[int]): Minimum distance between each line
            in line regions.
        min_line_pixels (Optional[int]): Minimum number of pixels along
            the height of an image to be considered as containing text.
            Defaults to 1.
        min_char_dist (Optional[int]): Minimum distance between each character
            in character regions. Defaults to 0.
        min_char_pixels (Optional[int]): Minimum number of pixels along a
            column in a line region to be considered as containing text.
            Defaults to 1.
        spell_check (Optional[bool]): Use spell check if True.
            Defaults to False.

    Returns:
        str: The string extracted from the image.
    """
    line_regs = line_regions(img, bg_thresh=bg_thresh, min_dist=min_line_dist,
                             min_pixels=min_line_pixels)
    char_regs = character_regions(
        img, line_regs, bg_thresh=bg_thresh, min_dist=min_char_dist,
        min_pixels=min_char_pixels)
    text = get_text_from_regions(img, line_regs, char_regs, classifier,
                                 resize=resize, **kwargs)

    if spell_check:
        text = " ".join(correct(word) for word in text.split())

    return text


def save_images(img, line_regs, char_regs, save_as, dest_dir):
    """
    Generate training data of individual characters from a grid of characters
    and save them into a destination directory.

    Args:
        img (numpy.ndarray): Base image.
        line_regs (list[tuple[int, int]]): Line regions from image.
        char_regs (list[list[tuple[int, int]]]): Character regions in each
            line in the image.
        save_as (list[str]): List of labels to save each row of like
            characters in the image.
        dest_dir (str): Directory to save each sample to.
    """
    assert len(line_regs) == len(save_as) == len(char_regs), \
        (("The number of line regions ({}), labels to save as ({}), and rows "
          "of character regions ({}) must all be the same.")
         .format(len(line_regs), len(save_as), len(char_regs)))

    import os
    import errno

    # Create each nested dir
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
    """Just split a string of labels into a list."""
    return labels.split(",")


def extraction_argument_parser(parser=None):
    """
    Generate an ArgumentParser with flags set for extracting text from an
    image.
    """
    if parser is None:
        from argparse import ArgumentParser
        parser = ArgumentParser()

    parser.add_argument("filename", help="File to extract text from.")
    parser.add_argument("--min_line_dist", type=int, default=0,
                        help="Minimum distance between each line. Defaults to "
                        "%(default)d.")
    parser.add_argument("--min_char_dist", type=int, default=0,
                        help="Minimum distance between each character. "
                        "Defaults to %(default)d.")
    parser.add_argument("--min_line_pixels", type=int, default=1,
                        help="Minimum number of dark pixels along the height  "
                        "of an image for that line to be considered as having "
                        "text. Defaults to %(default)d.")
    parser.add_argument("--min_char_pixels", type=int, default=1,
                        help="Minimum number of dark pixels along the width  "
                        "of a line for that line to be considered as having "
                        "text. Defaults to %(default)d.")
    parser.add_argument("--width", type=int, default=20,
                        help="Width to resize each sample image when running "
                        "through classifier. Defaults to %(default)d.")
    parser.add_argument("--height", type=int, default=20,
                        help="Height to resize each sample image when running "
                        "through classifier. Defaults to %(default)d.")
    parser.add_argument("-t", "--thresh", type=int,
                        help="Background threshold up to which a pixel is "
                        "considered color. For example, if --thresh is "
                        "200, then grayscaled pixels with a value of at most "
                        "200 are considered colored.")
    parser.add_argument("-p", "--pickle",
                        help="Pickle file containing a saved classifier to "
                        "use to classify text in the image.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Logging verbosity. More verbose means more "
                        "logging info.")
    parser.add_argument("--resize", type=float, default=1,
                        help="Base image resize ratio. If the given image is "
                        "too big, resizing the image to a smaller aspect "
                        "ratio will make this program run faster. "
                        "Defaults to %(default)0.1f.")
    parser.add_argument("--spell_check", action="store_true", default=False,
                        help="Try to improve extracted text by spell checking "
                        "each word. "
                        "Defaults to %(default)r.")

    # Set logging verbosity
    args = parser.parse_args()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s",
                        stream=sys.stderr)
    if args.verbose == 1:
        LOGGER.setLevel(logging.INFO)
    elif args.verbose == 2:
        LOGGER.setLevel(logging.DEBUG)

    return parser


def region_argument_parser(parser=None):
    """
    Generate an ArgumentParser with flags set for analyzing the regions of
    an image.
    """
    if parser is None:
        from argparse import ArgumentParser
        parser = ArgumentParser()

    parser.add_argument("-s", "--save_dir",
                        help="Directory to save training data into if the "
                        "labels are provided.")
    parser.add_argument("-l", "--labels", type=split_labels, default=[],
                        help="Labels for each sample, separated by commas.")

    return parser.parse_args()


def main():
    args = region_argument_parser(extraction_argument_parser())

    # Resize the image
    resize_ratio = args.resize
    img = cv2.imread(args.filename, 0)
    if resize_ratio != 1:
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    LOGGER.info("Checking image {}".format(args.filename))
    LOGGER.info("resized image shape: {}".format(img.shape))

    # Get average background color
    h, w = img.shape[:2]
    ravel = img.ravel()
    fig1 = plt.figure()
    plt.hist(ravel, 256, [0, 256])
    avg = np.mean(ravel)
    std = np.std(ravel)
    LOGGER.info("Image statistics")
    LOGGER.info("mean: {}".format(avg))
    LOGGER.info("std: {}".format(std))
    LOGGER.info("median: {}".format(np.median(ravel)))
    thresh = args.thresh
    if thresh is None:
        thresh = default_background_threshold(img)
    img = grayscale_to_black_and_white(img, thresh)
    LOGGER.info("background threshold: {}".format(thresh))
    fig1.show()

    # Find colored pixels
    text_pixels = colored_pixels(img, thresh)
    fig2 = plt.figure()
    plt.imshow(img, cmap='gray')
    xs, ys = zip(*text_pixels)
    assert len(xs) == len(ys)
    assert max(xs) <= w
    assert max(ys) <= h

    plt.scatter(xs, ys, marker=".", color="r")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    fig2.show()

    # Draw line boundaries
    line_positions = line_regions(img, min_dist=args.min_line_dist,
                                    bg_thresh=thresh,
                                    min_pixels=args.min_line_pixels)
    for pos in line_positions:
        plt.plot([0, w], [pos[0], pos[0]], color="b")
        plt.plot([0, w], [pos[1], pos[1]], color="b")

    # Draw character boundaries
    char_regions = character_regions(
        img, line_positions, bg_thresh=thresh, min_dist=args.min_char_dist,
        min_pixels=args.min_char_pixels)
    assert len(char_regions) == len(line_positions)
    for i, char_region in enumerate(char_regions):
        starty, endy = line_positions[i]
        for startx, endx in char_region:
            plt.plot([startx, startx], [starty, endy], color="b")
            plt.plot([endx, endx], [starty, endy], color="b")

    if args.save_dir:
        if not args.labels:
            raise RuntimeError(
                "If you are saving the characters as sample/training data, "
                "you must provided a comma-separated string of labels to "
                "save each sample as.")
        save_images(img, line_positions, char_regions, args.labels,
                    args.save_dir)
    else:
        # Keep figures alive
        raw_input("Press 'Enter' to close the current images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

