import numpy as np
import cv2
import os
import string
import time
from sklearn.neural_network import MLPClassifier
from random import randint

ALPHA_NUMERIC = string.digits + string.ascii_uppercase + string.ascii_lowercase


def showimg(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imgfile_to_grayscale(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def imgfile_to_grayscale2(filename, resize=20):
    assert os.path.isfile(filename)
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (resize, resize))
    return img


def load_digits():
    """Load training data from digits.png"""
    gray = imgfile_to_grayscale("digits.png")

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Training data
    X = [np.reshape(x[y][x_], (400, )).astype(np.float32) / 256 for x_ in xrange(100) for y in xrange(50)]

    # Expected
    y = [y for y in xrange(10) for x_ in xrange(len(X) / 10)]
    assert len(X) == len(y)

    return X, y


def test_digits(X, y, clf, test=100):
    correct = 0
    for i in xrange(test):
        num = randint(0, 9)
        i = len(X) / 10 * num + randint(0, len(X) / 10 - 1)
        print "expected: ", y[i]
        assert num == y[i]
        prediction = clf.predict([X[i]])[0]
        print "prediction: ", prediction
        if num == prediction:
            correct += 1
    print "% correct: ", correct * 100.0 / test


def load_english_hand(samples=55, base_dir="English/Hnd/Img", resize=30):
    """
    Load english handwritten characters  (using pc tablet) from
    http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

    Samples (55 of each; total 3410 samples)
    0-10:
        Numbers 0-9
    11-36:
        Uppercase
    37-62:
        Lowercase
    """
    start = time.time()
    X = [None] * 3410
    y = [c for c in ALPHA_NUMERIC for x in xrange(samples)]
    with open("English/Hnd/Img/all.txt~", "r") as samples:
        for i, sample in enumerate(samples):
            sample = sample.strip()
            x = imgfile_to_grayscale2(os.path.join(base_dir, sample), resize=resize)
            X[i] = np.array(np.reshape(x, (resize**2, )).astype(np.float32) / 256)
            print "Loaded:", i
    print "Loading training set:", time.time() - start, "seconds"
    return X, y


def test_chars(X, y, clf):
    correct = 0
    for i in xrange(len(X)):
        expected = ALPHA_NUMERIC[i / 55]
        print "expected: ", expected
        assert expected == y[i]
        prediction = clf.predict([X[i]])[0]
        print "prediction: ", prediction
        if expected == prediction:
            correct += 1
    print "% correct: ", correct * 100.0 / len(X)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("test_data", choices=("digits", "english"),
                        help="Test data to use.")

    return parser.parse_args()


def main():
    args = get_args()

    #hls = (400, 100)
    #hls = (100, 25)  # 94% on digits
    hls = (40, 20)

    if args.test_data == "digits":
        X, y = load_digits()
    else:
        X, y = load_english_hand()

    clf = MLPClassifier(hidden_layer_sizes=hls, verbose=True,
                        activation="tanh", max_iter=1000, algorithm="sgd",
                        learning_rate_init=0.01)
    clf.fit(X, y)

    if args.test_data == "digits":
        test_digits(X, y, clf)
    else:
        test_chars(X, y, clf)

    return 0


if __name__ == "__main__":
    main()

