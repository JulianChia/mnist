#!/usr/bin/env python3
#
# mnist_from_keras.py
#
#     https://github.com/JulianChia/mnist
#
# Copyright (C) 2025 Julian Chia
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
from urllib.request import urlretrieve
import numpy as np
from dataklasses import dataklass


URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'  # mnist datasets


@dataklass
class Mnist:
    train_images: np.array
    train_labels: np.array
    test_images: np.array
    test_labels: np.array


def set_MNIST_dir(file_parent_path: Path) -> Path:
    """Return Path to MNIST dataset"""
    if not file_parent_path:  # Set dir to current directory / mnist_from_keras
        return Path(__file__).parent.absolute() / 'mnist_from_keras'
    else:  # Set dir to given path / mnist_from_keras
        return Path(file_parent_path) / 'mnist_from_keras'


def download_MNIST_datasets_from_Keras(file_parent_path: Path) -> None:
    """Download mnist.npz file if missing."""
    npz_path = file_parent_path / 'mnist.npz'
    if npz_path.exists():
        print(f'{npz_path} exists. No need to download.')
    else:
        # Download MNIST files from Keras's Google Cloud Storage
        print(f'Downloading "mnist.npz" to {file_parent_path}... ', end='')
        urlretrieve(URL, npz_path)
        print(f'Completed!')


def configure_images(pixels: np.ndarray, normalise: bool, flatten: bool):
    """Return MNIST arrays with normalised and/or flatten values, if needed.
    By default, MNIST pixel values range from 0 (i.e. white) to 255 (i.e. black)
    and each image dimensions 28x28=784 pixels."""
    nimages = pixels.shape[0]
    nrows = pixels.shape[1]
    ncols = pixels.shape[2]

    if normalise:
        print(f'\n- images values range from 0.0(white) to 1.0(black).')
        pixels = pixels / 255
    else:
        print(f'\n- images values range from 0(white) to 255(black).')

    if flatten:
        print(f'- images shape is {nimages}x{nrows * ncols}.')
        return pixels.reshape(nimages, nrows * ncols)
    else:
        print(f'- images shape is {nimages}x{nrows}x{ncols}.')
        return pixels


def onehot_encoding(labels):
    """Return a 2D numpy array where only the element for the correct label
    is 1 and other elements are 0.

    Args:
     labels - 1D np.array : MNIST labels
    """
    rows = labels.size
    cols = labels.max() + 1
    onehot_labels = np.zeros((rows, cols), dtype='uint8')
    onehot_labels[np.arange(rows), labels] = 1
    return onehot_labels


def configure_labels(labels: np.ndarray, onehot: bool):
    """Return labels loaded locally."""
    # MNIST labels values range from 0 to 9.
    if onehot:
        labels = onehot_encoding(labels)
        print(f'- labels contain "onehot" values, shape: {labels.shape}.')
    else:
        print(f'- labels is 1D numpy array with uint8 values.')
    return labels


def load_MNIST(path=None, normalise=True, flatten=True, onehot=True):
    """Function to download, extract and configure MNIST train_images,
    train_labels, test_images and test_labels into dataklass objections for
    deep learning.

    dataklass from https://github.com/dabeaz/dataklasses

    Kwargs:
     path - str: MNIST datasets directory. Default to current directory/mnist_from_lecun.
                 Create if nonexistant. Download any missing MNIST files.
     normalise - boolean: True  -> pixels RGB values [0,255] divided by 255.
                          False -> pixels RGB values [0,255].
     flatten   - boolean: True  -> pixels of all images stored as 2D numpy array.
                          False -> pixels of all images stored as 3D numpy array.
     onehot    - boolean: True  -> labels stored as one-hot encoded numpy array.
                          False -> labels values used.

    Returns:
        A dataklass called 'Mnist' with numpy.ndarray attributes called
        'train_images', 'train_labels', 'test_images' and 'test_labels'.

        If normalise, the dtype of Mnist.train_images and Mnist.test_images are
        numpy.float64, else they will be numpy.uint8'

        if flatten, the shape of Mnist.train_images and Mnist.test_images is
        (60000, 784) and (10000, 784), respectively, else they will be
        (60000, 28, 28) and (10000, 28, 28), respectively.

        if onehot, the shape of Mnist.train_labels and Mnist.test_labels are
        (60000, 10) and (10000, 10), respectively, else they will be
        (60000,) and (10000,), respectively.
    """
    # Create MNIST path if it doesn't exist and download MNIST dataset if it
    # does not exist.
    mnist_path = set_MNIST_dir(path)
    try:
        mnist_path.mkdir(mode=0o777, parents=False, exist_ok=False)
    except FileExistsError:
        print(f'{mnist_path} exists. No need to create.')
    else:
        print(f'{mnist_path} is created.')
    finally:
        # Download MNIST_datasets if missing
        download_MNIST_datasets_from_Keras(mnist_path)

    # Load the .npz file
    mnist = np.load(mnist_path / 'mnist.npz')

    # Create the Mnist dataklass with the desired configurations
    return Mnist(configure_images(mnist['x_train'], normalise, flatten),
                 configure_labels(mnist['y_train'], onehot),
                 configure_images(mnist['x_test'], normalise, flatten),
                 configure_labels(mnist['y_test'], onehot)
                 )


if __name__ == "__main__":
    mdb = load_MNIST(path=None, normalise=True, flatten=True, onehot=True)  # default
    # mdb = load_MNIST()
    print(f'\nmdb is a {type(mdb)}')

    # You access it attributes as you would a Python class.  For example, by typing
    #   mdb.train_images, mdb.train_labels, mdb.test_images and mdb.test_labels.
    #
    # Below shows how to access the first training image and label:
    image0 = mdb.train_images[0]  # get first training image
    print(f'{image0.shape=}')
    print(f'{type(image0[0])=}')
    label0 = mdb.train_labels[0]  # get first training label
    print(f'{label0=}')
    print(f'{type(label0[0])=}')
