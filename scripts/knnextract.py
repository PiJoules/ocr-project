#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import logging

from ocr.extract import extraction_argument_parser, get_text
from ocr.classifiers.knn import KNN

LOGGER = logging.getLogger(__name__)


def get_args():
    """Add custom knn arguments."""
    parser = extraction_argument_parser()
    parser.add_argument("-k", "--knearest", type=int, default=5,
                        help="When using a knn classifier, this is the number "
                        "of neighbors to check. "
                        "Defaults to %(default)d.")
    return parser.parse_args()


def main():
    args = get_args()

    # Resize the image
    resize_ratio = args.resize
    img = cv2.imread(args.filename, 0)
    if resize_ratio != 1:
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    LOGGER.info("Checking image {}".format(args.filename))
    LOGGER.info("resized image shape: {}".format(img.shape))

    clf = KNN.from_file(args.pickle)
    LOGGER.info("Running knn classifier {}".format(args.pickle))
    resize = (args.width, args.height)
    text = get_text(img, clf, resize=resize, k=args.knearest,
                    bg_thresh=args.thresh, spell_check=args.spell_check,
                    min_char_dist=args.min_char_dist,
                    min_line_dist=args.min_line_dist,
                    min_char_pixels=args.min_char_pixels,
                    min_line_pixels=args.min_line_pixels)
    print(text)

    return 0


if __name__ == "__main__":
    main()

