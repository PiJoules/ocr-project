import numpy as np
import cv2
import string
import os

from matplotlib import pyplot as plt
from cropimage import *

ALPHA_NUMERIC = string.digits + string.ascii_uppercase + string.ascii_lowercase
TS_FORMAT = "%Y%m%d_%H%M%S"


def load_digits():
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    return train, train_labels, test, test_labels


def split_data(X, y, percentage, classes):
    """Split the data set into training and test data."""
    test_X = []
    test_y = []
    train_X = []
    train_y = []
    for i in xrange(classes):
        start_idx = len(y) / classes * i
        split_idx = int(start_idx + len(y) / classes * percentage)
        end_idx = len(y) / classes * (i + 1)
        train_X += X[start_idx:split_idx]
        train_y += y[start_idx:split_idx]
        test_X += X[split_idx:end_idx]
        test_y += y[split_idx:end_idx]
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def imgfile_to_grayscale(filename, resize=None):
    assert os.path.isfile(filename)
    img = cv2.imread(filename, 0)
    if resize:
        #trimmed = trimmed_image(img)
        #img = pad_and_resize(trimmed, resize[0], resize[1])
        img = cv2.resize(img, resize)
    return img


def load_shrinked_imgs(dirname, dim=20, samples=55, retain=0.8, classes=62):
    X = [None] * samples * classes
    y = [np.array([ord(c)]) for c in ALPHA_NUMERIC for x in xrange(samples)]
    i = 0
    for sample in os.listdir(dirname):
        if not sample.startswith("Sample"):
            continue
        dirpath = os.path.join(dirname, sample)
        for j, imgname in enumerate(os.listdir(dirpath)):
            if j >= samples:
                break
            filepath = os.path.join(dirpath, imgname)
            x = imgfile_to_grayscale(filepath)
            X[i] = np.array(np.reshape(x, (dim**2, ))).astype(np.float32)
            print "Loaded:", filepath, " as", y[i]
            i += 1

    return split_data(X, y, retain, classes)


def load_english_hand(samples=55, base_dir="English/Hnd/Img", classes=62,
                      resize=(30, 30), retain=0.8):
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
    X = [None] * samples * classes
    y = [np.array([ord(c)]) for c in ALPHA_NUMERIC for x in xrange(samples)]
    with open(os.path.join(base_dir, "all.txt~"), "r") as samples:
        for i, sample in enumerate(samples):
            sample = sample.strip()
            x = imgfile_to_grayscale(os.path.join(base_dir, sample), resize=resize)
            X[i] = np.array(np.reshape(x, (resize[0]*resize[1], ))).astype(np.float32)
            print "Loaded:", i+1
    return split_data(X, y, retain, classes)


def save_classifier(clf, filename="classifier.{timestamp}.p"):
    import pickle
    from datetime import datetime

    timestamp = datetime.now().strftime(TS_FORMAT)
    filename = filename.format(timestamp=timestamp)
    with open(filename, "wb") as save_file:
        pickle.dump(clf, save_file, protocol=-1)


class KNearest2(object):
    def __init__(self, clf):
        self.__clf = clf

    def save_classifier(self, filename="classifier.{timestamp}.p"):
        import pickle
        from datetime import datetime

        timestamp = datetime.now().strftime(TS_FORMAT)
        filename = filename.format(timestamp=timestamp)
        with open(filename, "wb") as save_file:
            pickle.dump(self, save_file)


def main():
    #train, train_labels, test, test_labels = load_digits()
    #train, train_labels, test, test_labels = load_shrinked_imgs("shrinked20")
    #train, train_labels, test, test_labels = load_english_hand()

    # Initiate kNN, train the data, then test it with test data for k=1
    x = cv2.ml.KNearest_create()
    save_classifier(x)
    #knn = cv2.KNearest()
    #save_classifier(knn)
    #knn.train(train, train_labels)
    #ret,result,neighbours,dist = knn.find_nearest(test, k=10)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy

if __name__ == "__main__":
    main()

