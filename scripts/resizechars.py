#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import os


def resize_imgs(resize=20, out_dir="shrinked", base_dir="English/Hnd/Img",
              overwrite=False):
    out_dir += str(resize)
    with open(os.path.join(base_dir, "all.txt~"), "r") as samples:
        for i, sample in enumerate(samples):
            sample = sample.strip()
            sample_file = os.path.join(base_dir, sample)
            sample_dir, sample_img = sample.split("/")
            mkdir(os.path.join(out_dir, sample_dir))
            save_file = os.path.join(out_dir, sample_dir, sample_img)

            if os.path.isfile(save_file) and not overwrite:
                continue
            x = imgfile_to_grayscale2(sample_file, resize=resize)

            cv2.imwrite(save_file, x)

            print("Shrunk", i, "to", save_file)

