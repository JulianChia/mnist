# mnist.py
![Title](MNIST.png)
This python module provides a simple to use function to download and extract the MNIST database of handwritten digits that is provided by http://yann.lecun.com/exdb/mnist/.

**Function:**

    load_MNIST(path=None, normalise=True, flatten=True, onehot=True)

_kwarg:_ 

    path - str: MNIST datasets directory. Default to current directory/MNIST.
                Create if nonexistant. Download any missing MNIST files.
    normalise - boolean: yes -> pixel RGB values [0,255] divided by 255.
                         no  -> pixel RGB values [0,255].
    flatten   - boolean: yes -> pixels of all images stored as 2D numpy array.
                         no  -> pixels of all images stored as 3D numpy array.
    onehot    - boolean: yes -> labels stored as one-hot encoded numpy array.
                         no  -> labels values used.

_Returns a nested dictionary:_

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
                     if onehot,   labels.shape = (60000, 10)
                     else,        labels.shape = (60000,)
      test_images = MNISTimages(magic_number=2051, nimages=10000, nrows=28,
                                ncols=28, pixels=np.array())
                    if normalise, pixelsdtype='float32'
                    else,         pixels dtype='uint8'
                    if flatten,   pixels.shape = (10000, 784)
                    else,         pixels.shape = (10000, 28, 28)
      test_labels = MNISTlabels(magic_number=2049, nlabels=10000,
                                labels=np.array() dtype='uint8')
                    if onehot,   labels.shape = (10000, 10)
                    else,        labels.shape = (10000,)

*Remarks:*

`MNISTimages()` and `MNISTlabels()` are [dataklass objects](https://github.com/dabeaz/dataklasses). On my system, they performed ~25x faster than python3 built-in [dataclass objects](https://docs.python.org/3/library/dataclasses.html) and 5x faster than [namedtuple](https://docs.python.org/3/library/collections.html?highlight=namedtuple#collections.namedtuple). 

# How to use?

    from mnist import load_MNIST           # Import function from module
    mdb = load_MNIST()                     # Get MNIST database using default settings
    train_images = mdb['train']['pixels']  # A 60000x784 numpy array with float32 values    
    train_labels = mdb['train']['labels']  # A 60000x10 numpy array with uint8 values
    test_images = mdb['test']['pixels']   # A 10000x784 numpy array with float32 values    
    test_labels = mdb['test']['labels']   # A 10000x10 numpy array with uint8 values