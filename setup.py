# setup.py for HyVR
# -----------------


import re
from os import path
from shutil import copyfile

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.install import install
from setuptools.command.develop import develop

setup(
    name="hyvr",
    version="0.1.0",
    description="sedimentary structures virtual reality generator",
    url="https://github.com/vcantarella/hyvr",
    author="Vitor Cantarella",
    license="MIT",
    packages=["hyvr"],
    install_requires=["numpy","scipy","numba"],
    keywords=['hydrogeology', 'sediment', 'simulator'],
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Opeating System:: POSIX :: Linux",
        "Programming Language :: Python 3",
    ]
)
