#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


def long_description():
    with open("README.md", "r") as readme:
        return readme.read()


def packages():
    return find_packages(include=["ocr*", "scripts*"])


def install_requires():
    with open("requirements.txt", "r") as requirements:
        return requirements.readlines()


setup(
    name="ocr",
    version="0.0.1",
    description="OCR development and testing.",
    long_description=long_description(),
    url="https://github.com/PiJoules/ocr-project",
    author="Leonard Chan",
    author_email="lchan1994@yahoo.com",
    license="Unlicense",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    package_data={
        "data": ["*"]
    },
    keywords="ocr,knn,mlp",
    packages=packages(),
    install_requires=install_requires(),
    entry_points={
        "console_scripts": [
            "regions=ocr.extract:main",
            "knn-create=ocr.classifiers.knn:main",
            "knn-extract=scripts.knnextract:main",
            "mlp-create=ocr.classifiers.mlp:main",
            "mlp-extract=scripts.mlpextract:main",
        ]
    }
)

