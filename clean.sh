#!/usr/bin/env bash

rm -rf *.egg-info
find ocr/ -name '*.pyc' -exec rm {} \;
rm -rf build/ dist/
rm -rf tests/*.pyc scripts/*.pyc
