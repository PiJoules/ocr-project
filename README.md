# OCR Project
The purpose of this repo is to construct an ocr for detecting keywords from
people's handwriting to make their notes searchable.
Whole sentences do not need to be fully extracted, as long as a few unique
keywords are extracted such that the extracted set of words are seo-able.


## Environment
This package was developed in the following environment.
- Python 2.7.9
- Mac OSX 10.11.4, Ubuntu 14.04
- pip 8.1.1


## Dependencies
This package is dependant on the packages provided in requirements.txt.
- numpy
- scipy
- matplotlib
- cv2

Optional development packages to install are provided in dev-requirements.txt.
Precompiling the pure code with cython can allow for an extra boost in
performance.

### OpenCV
This package is heavily dependant on opencv, so be sure to install it first
before installing the cv2 package in requirements.txt.

- [Installing on Ubuntu/Debian](http://milq.github.io/install-opencv-ubuntu-debian/)
- [Installing on Mac OS](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/)

### Scikit Learn
If using the multilayer perceptron, the latest dev version of scikit learn
(0.18.dev0) is required since it already contains an implementation of a
multilayer perceptron that this package uses.

At the time of writing this, version 0.17.0 is provided on pypi, so this dev
version cannot be installed via pip and will have to be installed by cloning
the [repo](https://github.com/scikit-learn/scikit-learn) and just following
the instructions in the repo for installing it.

### Troubleshooting
If you have trouble installing numpy, scipy, or cv2, try installing them
individually instead of together in the requirements.txt.

If you have trouble installing them in a virtualenv, try installing them on
your global pip instead.


## Installing
After cloning the repo, just run the following:

```sh
$ python setup.py develop  # Create executables
```

To make the ocr packageglobally availale on your computer via pip:
```sh
$ python setup.py bdist_wheel  # Create binary distribution
$ wheel install dist/ocr_testbed-{version}-py{2/3}-none-any.whl  # Install created wheel
$ wheel install-scripts ocr-testbed  # Enable package scripts
````
The first command with `develop` essentially does the same as the following
three commands, but these commands allow for the package to be available
globally through your computer in pip space like any other package installed
via pip.


## Usage
The package can be used to create classifiers and use these classifiers to
classify characters extracted from an image.

Before extracting text from an image, you will need to create a classifier to classify
each character extracted from the image.

### K Nearest Neighbors
Create a knn classifier using handwritten training data in the given dir and save it
in a file called classifier.p.
```sh
$ knn-create -t handwritten -d data/training/handwritten -s classifier.p
53.2258064516 %  # Success rate on testing data
```

Extract text from the given image using the classifier we just made. Resize the image
to reduce computation time.
```sh
$ knn-extract data/test/sample3.jpg -p classifier.p --resize 0.25
6hhVL tS L mUYh kYttrYYr 6b h qYhcY  # Yeah, this isn't very good output
```

### Multilayer Perceptron
Create an mlp classifier using shrinked, handwritten training data in the given
dir and save it in a file called classifier.p.
```sh
$ mlp-create -t shrinked -d data/shrinked20x20/ -s classifier.p
63.3431085044 %  # Success rate on testing data
```

Extract text from the given image using the classifier we just made. Resize the image
to reduce computation time.
```sh
$ mlp-extract data/test/sample3.jpg -p classifier.p --resize 0.25
iYYij jY N CjYj iYjV1Y1L 1j Y YYLG1  # Yeah, this isn't very good output
```


## Development
To develop/test other classifiers, you would just need to create a classifier
wrapper from `ocr.classifiers.base.Classifier` and create an extraction
script that uses this classifier to get text from an image.

Example usages of this are in the KNN and MLP Classifiers in ocr/classifiers/
and examples of implementing the extraction script are in scripts/.


## Results
The package functionally works in that it can create and train classifiers, and use
these classifiers to extract text from images, but this package does not work in that
the correct text is extracted. The following are ways to improve performance:
- Better sample data
- Better features
  - Only the individual grayscale pixel values are used as featues. Alternative
    features that should be experimented with include:
    - Number of edges on each character
    - Number of circles on each character
    - Projecting each character onto the x and y axis and using the distributions
      along each axis as featues.
- Smarter character region finding in an image.

For now, this package works best as a testbed for experimenting with different
classifiers and features.


## TODO
- Update unit tests.
- Make presentation.
