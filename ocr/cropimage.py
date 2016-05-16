import cv2
import numpy as np


def trim_image(img, white_bg=True, thresh=255):
    """img is 2D rectangular grayscale image"""
    if not white_bg:
        thresh = 255 - thresh

    top_found = False
    bot_found = False
    left_found = False
    right_found = False

    top_ind = img.shape[1]
    bot_ind = 0
    left_ind = img.shape[0]
    right_ind = 0

    for r, row in enumerate(img):
        for c, pix in enumerate(row):
            if pix <= thresh:
                top_ind = min(top_ind, r)
                bot_ind = max(bot_ind, r)
                left_ind = min(left_ind, c)
                right_ind = max(right_ind, c)

    return (top_ind, bot_ind, left_ind, right_ind)


def trimmed_image(img, **kwargs):
    """
    Return the trimmed image.

    **kwargs: The same arguments passed into trim_image.
    """
    top, bot, left, right = trim_image(img, **kwargs)
    return img[top:bot + 1, left:right + 1]


def pad_and_resize(img, h, w, bg=255):
    # Pad
    ar_img = 1.0*img.shape[0]/img.shape[1]
    ar_des = 1.0*h/w
    if ar_img > ar_des:
        # Too tall, adjust cols keeping # of rows fixed
        cols_des = int(img.shape[0] / ar_des)
        # Arbitrarily padding more on right than left
        padded = np.lib.pad(img, ((0, 0), (cols_des / 2, (cols_des + 1) / 2)),
                            'constant', constant_values=((bg, bg), (bg, bg)))
    else:
        # Too wide, adjust rows keeping # of cols fixed
        rows_des = int(img.shape[1] * ar_des)
        # Arbitrarily padding more on bottom than top
        padded = np.lib.pad(img, ((rows_des / 2, (rows_des + 1) / 2), (0, 0)),
                            'constant', constant_values=((bg, bg), (bg, bg)))

    # Resize
    return cv2.resize(padded, (h, w))


if __name__ == "__main__":
    img = cv2.imread("j.jpg", 0)
    trimmed = trimmed_image(img, thresh=240)
    print trimmed.shape
    resized = pad_and_resize(trimmed, 30, 30)
    print resized.shape
    cv2.imwrite("jtrimmed.png", trimmed)
    cv2.imwrite("jresized.png", resized)
