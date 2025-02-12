#!/usr/bin/env python3
#
# mnist_from_lecun.py
#
#     https://github.com/JulianChia/mnist
#
# Copyright (C) 2022-2025 Julian Chia
#
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
import numpy as np
from dataklasses import dataklass
import requests


# URL = 'http://yann.lecun.com/exdb/mnist/'  # Obsolete
URL = 'https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/'
FILES = ['train-images-idx3-ubyte.gz',  # train_images
         'train-labels-idx1-ubyte.gz',  # train_labels
         't10k-images-idx3-ubyte.gz',  # test_images
         't10k-labels-idx1-ubyte.gz']  # test_labels


@dataklass
class Mnist:
    train_images: np.array
    train_labels: np.array
    test_images: np.array
    test_labels: np.array


def set_MNIST_dir(file_parent_path: Path) -> Path:
    if not file_parent_path:  # Set dir to current directory / mnist_from_lecun
        return Path(__file__).parent.absolute() / 'mnist_from_lecun'
    else:  # Set dir to given path / mnist_from_lecun
        return Path(file_parent_path) / 'mnist_from_lecun'


def download_MNIST_datasets_from_LeCun(file_parent_path: Path) -> None:
    """Download any missing files."""
    for file in FILES:
        filepath = file_parent_path / file
        if not filepath.exists():
            print(f'Downloading {file} to {file_parent_path} ... ', end='')

            # opener = build_opener()
            # install_opener(opener)
            # urlretrieve(URL + file, filepath)

            with open(filepath, "wb") as f:
                r = requests.get(URL + file)
                f.write(r.content)
            print(f'Completed!')
        else:
            print(f'{file} exists. No need to download.')


def get_int(byte_value: bytes):
    """Function to convert byte to int, byteorder is big as MSB is at start."""
    return int.from_bytes(byte_value, "big")


def extract_images(filepath: Path, normalise: bool, flatten: bool):
    """Return image dataset with desired configuration."""
    with gzip.open(filepath, 'rb') as f:
        contents = f.read()
        # First 16 bytes are magic_number, nimages, nrows, ncols
        # magic_number = get_int(contents[0:4])
        nimages = get_int(contents[4:8])
        nrows = get_int(contents[8:12])
        ncols = get_int(contents[12:16])

        # Subsequent bytes are pixels values of images.
        # MNIST pixels are organized row-wise. Pixel values are 0 to 255.
        # - 0 means background (white), 255 means foreground (black).
        # - Each image dimensions 28x28=784 pixels
        if normalise:
            print(f'\n- images values range from 0.0(white) to 1.0(black).')
            pixels = np.frombuffer(contents, dtype='B', offset=16).astype('f') / 255
        else:
            print(f'\n- images values range from 0(white) to 255(black).')
            pixels = np.frombuffer(contents, dtype='uint8', offset=16)

        if flatten:
            print(f'- images shape is {nimages}x{nrows * ncols}.')
            return pixels.reshape(nimages, nrows * ncols)
        else:
            print(f'- images shape is {nimages}x{nrows}x{ncols}.')
            return pixels.reshape(nimages, nrows, ncols)


def onehot_encoding(labels: np.ndarray):
    """Return a 2D numpy array where only the element for the correct label
    is 1 and other elements are 0.

    Args:
     labels - 1D np.array : MNIST labels
    """
    rows = labels.size
    cols = labels.max() + 1
    onehot = np.zeros((rows, cols), dtype='uint8')
    onehot[:, labels] = 1
    return onehot


def extract_labels(filepath, onehot):
    """Return label dataset with desired configuration."""
    with gzip.open(filepath) as f:
        contents = f.read()
        # First 8 bytes are magic_number, nlabels
        # magic_number = get_int(contents[0:4])
        # nlabels = get_int(contents[4:8])
        # Subsequent bytes are value of labels.
        # MNIST labels are organized row-wise. Labels values are 0 to 9.
        labels = np.frombuffer(contents, 'B', offset=8)
        if onehot:
            print('- labels is a 2D numpy array with "onehot" values.')
            labels = onehot_encoding(labels)
        else:
            print('- labels is a 1D numpy array with uint8 values.')
            labels = labels.astype('uint8')
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
        download_MNIST_datasets_from_LeCun(mnist_path)

    # Create the Mnist dataklass with the desired configurations
    return Mnist(extract_images(mnist_path / FILES[0], normalise, flatten),
                 extract_labels(mnist_path / FILES[1], onehot),
                 extract_images(mnist_path / FILES[2], normalise, flatten),
                 extract_labels(mnist_path / FILES[3], onehot))


if __name__ == "__main__":
    # mdb = load_MNIST(path=None, normalise=False, flatten=True, onehot=False)
    mdb = load_MNIST()
    print(f'\nmdb = {type(mdb)}')

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
