"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # Read Iamge File
    with gzip.open(image_filename, "rb") as images_file:
        magic_num = int.from_bytes(images_file.read(4), "big")
        images_num = int.from_bytes(images_file.read(4), "big")
        rows_num = int.from_bytes(images_file.read(4), "big")
        cols_num = int.from_bytes(images_file.read(4), "big")

        images_matrix = np.frombuffer(images_file.read(), dtype=np.uint8)
        images_matrix = images_matrix.astype(np.float32)
        images_matrix = images_matrix.reshape((images_num, rows_num * cols_num))
        images_matrix = images_matrix / 255

    # Read Label File
    with gzip.open(label_filename, "rb") as labels_file:
        magic_num = int.from_bytes(labels_file.read(4), "big")
        labels_num = int.from_bytes(labels_file.read(4), "big")
        labels_array = np.frombuffer(labels_file.read(), dtype=np.uint8)

    return (images_matrix, labels_array)


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # Numerical Stability
    Z = Z - ndl.max(Z)
    z_log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    z_y = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(z_log_sum_exp - z_y) / np.float32(Z.shape[0])


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    for i in range(0, X.shape[0], batch):
        X_b = ndl.Tensor(X[i : i + batch])
        Z1 = ndl.relu(ndl.matmul(X_b, W1))
        Z2 = ndl.matmul(Z1, W2)
        Y = np.zeros(Z2.shape, np.float32)
        Y[np.arange(y[i : i + batch].size), y[i : i + batch]] = 1
        Y_b = ndl.Tensor(Y)
        loss = softmax_loss(Z2, Y_b)
        loss.backward()
        # Only update the data of W1 and W2 to avoid add
        # the reference to previous iteration's computation graph
        W1.data = W1.data - W1.grad.data * lr
        W2.data = W2.data - W2.grad.data * lr
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
