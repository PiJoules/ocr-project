#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


def long_description():
    with open("README.md", "r") as readme:
        return readme.read()


def packages():
    return find_packages(include=["ocr*"])


def install_requires():
    with open("requirements.txt", "r") as requirements:
        return requirements.readlines()


setup(
    name="ocr",
    version="0.0.1",
    description="{description}",
    long_description=long_description(),
    url="{url}",
    author="{author}",
    author_email="{author_email}",
    license="{license}",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    keywords="{keywords}",
    packages=packages(),
    install_requires=install_requires(),
    test_suite="tests",
    entry_points={
        "console_scripts": [
        ]
    }
)

