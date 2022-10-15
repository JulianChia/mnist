#!/usr/bin/env python3
#
# mnist.py
#
#     https://github.com/JulianChia/mnist
#
# Copyright (C) 2022 Julian Chia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = ['load_MNIST']

from pathlib import Path
import gzip
from urllib.request import build_opener, install_opener, urlretrieve
import numpy as np
from dataklasses import dataklass

@dataklass
class MNISTimages:
    magic_number: int
    nimages: int
    nrows: int
    ncols: int
    pixels: np.array


@dataklass
class MNISTlabels:
    magic_number: int
    nlabels: int
    labels: np.array


def load_MNIST(path=None, normalise=True, flatten=True, onehot=True):
    """Function to download and extract MNIST train_images, train_labels,
    test_images and test_labels into dataklass objections for deep learning.

    dataklass from https://github.com/dabeaz/dataklasses

    Args:
     path - str: MNIST datasets directory. Default to current directory/MNIST.
                 Create if nonexistant. Download any missing MNIST files.
     normalise - boolean: yes -> pixel RGB values [0,255] divided by 255.
                          no  -> pixel RGB values [0,255].
     flatten   - boolean: yes -> pixels of each image stored as 1D numpy array.
                          no  -> pixels of each image stored as 2D numpy array.
     onehot    - boolean: yes -> labels stored as one-hot encoded numpy array.
                          no  -> labels values used.

    Returns:
     {'train': {'images': train_images, 'labels': train_labels},
      'test': {'images': test_images, 'labels': test_labels}}
     where,
      train_images = MNISTimages(magic_number=2051, nimages=60000, nrows=28,
                                 ncols=28, pixels=np.array())
                     if normalise, pixels dtype='float32'
                     else,         pixels dtype='uint8'
                     if flatten,   pixels.shape = (60000, 784)
                     else,         pixels.shape = (60000, 28, 28)
      train_labels = MNISTlabels(magic_number=2049, nlabels=60000,
                                 labels=np.array() dtype='uint8')
                     if onehot,    labels.shape = (60000, 10)
                     else,         labels.shape = (60000,)
      test_images = MNISTimages(magic_number=2051, nimages=10000, nrows=28,
                                ncols=28, pixels=np.array())
                    if normalise,  pixels dtype='float32'
                    else,          pixels dtype='uint8'
                    if flatten,    pixels.shape = (10000, 784)
                    else,          pixels.shape = (10000, 28, 28)
      test_labels = MNISTlabels(magic_number=2049, nlabels=10000,
                                labels=np.array() dtype='uint8')
                    if onehot,     labels.shape = (10000, 10)
                    else,          labels.shape = (10000,)
    """
    def _set_MNIST_dir(file_parent_path):
        if not file_parent_path:  # Set dir to current directory / MNIST
            return Path(__file__).parent.absolute() / 'MNIST'
        else:  # Set dir to given path / MNIST
            return Path(file_parent_path) / 'MNIST'

    def _download_MNIST_datasets(url, files, file_parent_path):
        """Download any missing files."""
        for file in files:
            filepath = file_parent_path / file
            if not filepath.exists():
                print(f'Downloading {file} to {file_parent_path}... ', end='')
                opener = build_opener()
                install_opener(opener)
                urlretrieve(url + file, filepath)
                print(f'Completed!')
            else:
                print(f'{file} exists. No need to download.')

    def _get_int(byte_value):
        """Function to convert byte to int, byteorder is big as MSB is at
        start."""
        return int.from_bytes(byte_value, "big")

    def _normalise_or_not(contents):
        if normalise:
            print('- pixels value range from 0.0(white) to 1.0(black).')
            return np.frombuffer(contents, dtype='B', offset=16).astype('f')/255
        else:
            print('- pixels value range from 0(white) to 255(black).')
            return np.frombuffer(contents, dtype='uint8', offset=16)

    def _flatten_or_not(pixels, nimages, nrows, ncols):
        if flatten:
            print(f'- pixels is a numpy array of shape {nimages}x{nrows*ncols}.')
            return pixels.reshape(nimages, nrows * ncols)
        else:
            print(f'- pixels is a numpy array of shape {nimages}x{nrows}x{ncols}.')
            return pixels.reshape(nimages, nrows, ncols)

    def _onehot_encoding(labels):
        """Return a 2D numpy array where only the element for the correct label
        is 1 and other elements are 0.

        Args:
         labels - 1D np.array : MNIST labels
        """
        rows = labels.size
        cols = labels.max() + 1
        onehot = np.zeros((rows, cols), dtype='uint8')
        onehot[np.arange(rows), labels] = 1
        return onehot

    def _extract_images(filepath):
        """Return images loaded locally."""
        with gzip.open(filepath) as f:
            contents = f.read()
            # First 16 bytes are magic_number, nimages, nrows, ncols
            magic_number = _get_int(contents[0:4])
            nimages = _get_int(contents[4:8])
            nrows = _get_int(contents[8:12])
            ncols = _get_int(contents[12:16])
            # Subsequent bytes are pixels values of images.
            # MNIST pixels are organized row-wise. Pixel values are 0 to 255.
            # - 0 means background (white), 255 means foreground (black).
            # - Each image dimensions 28x28=784 pixels
            pixels = _normalise_or_not(contents)
            pixels = _flatten_or_not(pixels, nimages, nrows, ncols)
        return MNISTimages(magic_number, nimages, nrows, ncols, pixels)

    def _extract_labels(filepath):
        """Return labels loaded locally."""
        with gzip.open(filepath) as f:
            contents = f.read()
            # First 8 bytes are magic_number, nlabels
            magic_number = _get_int(contents[0:4])
            nlabels = _get_int(contents[4:8])
            # Subsequent bytes are value of labels.
            # MNIST labels are organized row-wise. Labels values are 0 to 9.
            labels = np.frombuffer(contents, 'B', offset=8)
            if onehot:
                print('- labels is a 2D numpy array with "onehot" values.')
                labels = _onehot_encoding(labels)
            else:
                print('- labels is a 1D numpy array with uint8 values.')
                labels = labels.astype('uint8')
            print(labels.shape)
        return MNISTlabels(magic_number, nlabels, labels)

    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',  # train_images
             'train-labels-idx1-ubyte.gz',  # train_labels
             't10k-images-idx3-ubyte.gz',   # test_images
             't10k-labels-idx1-ubyte.gz']   # test_labels
    # Create path if it doesn't exist and download MNIST datasets there if they
    #  do not exist.
    path = _set_MNIST_dir(path)
    try:
        path.mkdir(mode=0o777, parents=False, exist_ok=False)
    except FileExistsError:
        print(f'{path} exists. No need to create.')
    else:
        print(f'{path} is created.')
    finally:
        # Download any missing files
        _download_MNIST_datasets(url, files, path)
    # Extract datasets
    print('Train Images and Labels:')
    train_images = _extract_images(path / files[0])
    train_labels = _extract_labels(path / files[1])
    print('Test Images and Labels:')
    test_images = _extract_images(path / files[2])
    test_labels = _extract_labels(path / files[3])
    # Store extracted datasets in a dict
    train = {'images': train_images, 'labels': train_labels}
    test = {'images': test_images, 'labels': test_labels}
    return {'train': train, 'test': test}


if __name__ == "__main__":
    # mdb = load_MNIST(path=None, normalise=False, flatten=True, onehot=False)
    mdb = load_MNIST()
    print(f'mdb = {mdb}')
    print((mdb['train']['images'].pixels.shape))
    print((mdb['train']['labels'].labels.shape))
    print((mdb['test']['images'].pixels.shape))
    print((mdb['test']['labels'].labels.shape))
