import cv2
import numpy as np

"""
Crop and resize a grayscale image.
"""


def trim_image(img, white_bg=True, thresh=240):
    """
    Trim edges of image.

    Args:
        img (numpy.array): 2D rectangular grayscale image
    """
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
    """Resize the image, attempting to keep constant aspect ratio."""
    # Pad
    ar_img = 1.0*img.shape[0]/img.shape[1]
    ar_des = 1.0*h/w
    if ar_img > ar_des:
        # Too tall, adjust cols keeping # of rows fixed
        cols_des = int(img.shape[0] / ar_des)
        # Arbitrarily padding more on right than left
        padded = np.lib.pad(img, ((0, 0), (cols_des / 2, (cols_des + 1) / 2)),
                            'constant',
                            constant_values=np.array([(bg, bg), (bg, bg)]))
    else:
        # Too wide, adjust rows keeping # of cols fixed
        rows_des = int(img.shape[1] * ar_des)
        # Arbitrarily padding more on bottom than top
        padded = np.lib.pad(img, ((rows_des / 2, (rows_des + 1) / 2), (0, 0)),
                            'constant', constant_values=np.array([(bg, bg), (bg, bg)]))

    # Resize
    return cv2.resize(padded, (h, w))


def get_args():
    """Parse cmd line args."""
    from argparse import ArgumentParser
    parser = ArgumentParser("Trim, pad, and resize image.")

    parser.add_argument("imgfile", help="Image file to shrink and pad.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output filename.")
    parser.add_argument("-w", "--width", type=int, default=20,
                        help="Desired width.")
    parser.add_argument("-h", "--height", type=int, default=20,
                        help="Desired height.")

    return parser.parse_args()


def main():
    """Entry point."""
    args = get_args()

    img = cv2.imread(args.imgfile, 0)
    trimmed = trimmed_image(img)
    resized = pad_and_resize(trimmed, args.width, args.height)
    cv2.imwrite(args.output, resized)

    return 0


if __name__ == "__main__":
    main()

