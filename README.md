# OCR Project
The purpose of this repo is to construct an ocr for detecting keywords from people's handwriting to make their notes searchable.
Whole sentences do not need to be fully extracted, as long as a few unique keywords are extracted such that the extracted set of words
are seo-able.

## OCR
The optical character recognizer will be constructed from a multilayer perceptron for simplicity, and will need to classify 62 characters:
10 digits, 26 english lowercase letters, and 26 english uppercase letters.
The intended training data will be the person's handwriting.

