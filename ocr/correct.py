# -*- coding: utf-8 -*-

"""
Spell checker from:
    http://norvig.com/spell-correct.html
"""

import re
import collections
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
REFERENCE_FILE = os.path.join(DATA_DIR, "texts", "big.txt")


def words(text):
    """Get all words from the text."""
    return re.findall("[a-z]+", text.lower())


def train(features):
    """Get the frequency of each word."""
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


# Generate word frequency
NWORDS = train(words(file(REFERENCE_FILE).read()))
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def edits(word):
    """
    Edit each word by splitting the word, deleting, replacing, and inserting
    groups of characters, and transposing each sub-word.
    """
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in ALPHABET if b]
    inserts    = [a + c + b     for a, b in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)

def known_edits(word):
    """Get the words shown in NWORDS that can be derived from edits."""
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in NWORDS)

def known(words):
    """Get the words that appear in NWORDS."""
    return set(w for w in words if w in NWORDS)

def correct(word):
    """Get the best guess of the correct spelling of a word."""
    word = word.lower()
    candidates = known([word]) or known(edits(word)) or known_edits(word) or [word]
    return max(candidates, key=NWORDS.get)

