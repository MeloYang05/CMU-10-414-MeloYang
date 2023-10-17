"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -out_grad * a * (b ** (-2))
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        np_axes = list(range(len(a.shape)))
        if not self.axes is None:
            np_axes[self.axes[0]] = self.axes[1]
            np_axes[self.axes[1]] = self.axes[0]
        else:
            np_axes[len(a.shape) - 2] = len(a.shape) - 1
            np_axes[len(a.shape) - 1] = len(a.shape) - 2
        return numpy.transpose(a, np_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return numpy.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad.reshape(a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return numpy.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        a_shape = list(a.shape)
        grad_shape = list(out_grad.shape)
        if len(grad_shape) > len(a_shape):
            a_shape += [1] * (len(grad_shape) - len(a_shape))
        axes = []
        for i in range(len(a_shape)):
            if grad_shape[i] / a_shape[i] > 1 and grad_shape[i] % a_shape[i] == 0:
                axes.append(i)
        return out_grad.sum(tuple(axes)).reshape(a.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return numpy.sum(a, self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        if self.axes is None:
            return out_grad.broadcast_to(a.shape)
        else:
            shape = list(a.shape)
            for i in range(len(shape)):
                if i in self.axes:
                    shape[i] = 1
            return out_grad.reshape(tuple(shape)).broadcast_to(a.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return numpy.matmul(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad.matmul(b.transpose())
        grad_b = a.transpose().matmul(out_grad)
        if len(a.shape) > len(b.shape):
            grad_b = grad_b.sum(tuple(range(len(a.shape) - len(b.shape))))
        elif len(b.shape) > len(a.shape):
            grad_a = grad_a.sum(tuple(range(len(b.shape) - len(a.shape))))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return numpy.negative(a)

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return numpy.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad / a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return numpy.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)


class Gt(TensorOp):
    def compute(self, a, b):
        return a > b

    def gradient(self, out_grad, node):
        return out_grad * 0


def gt(a):
    return Gt()(a)


class GtScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a > self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * 0


def gt_scalar(a, scalar):
    return GtScalar(scalar)(a)


class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(0, a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * (a > 0)


def relu(a):
    return ReLU()(a)
