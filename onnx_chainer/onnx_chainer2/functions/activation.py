from __future__ import print_function


import chainer
import chainer.functions as F
from onnx_chainer.onnx_chainer2 import converter


@converter.generic_converter(1)
def clipped_relu(gb, x, z):
    return gb.Clip([x], min=0.0, max=float(z))


@converter.generic_converter(1)
def elu(gb, x, alpha):
    return gb.Elu([x], alpha=float(alpha))


@converter.generic_converter(1)
def leaky_relu(gb, x, slope):
    return gb.LeakyRelu([x], alpha=float(slope))


@converter.generic_converter(1)
def log_softmax(gb, x, axis):
    assert axis == 1, 'LogSoftmax with axis != 1 is not supported yet.'
    return gb.LogSoftmax([x], axis=int(axis))


@converter.generic_converter(1)
def softmax(gb, x, axis):
    assert axis == 1, 'Softmax with axis != 1 is not supported yet.'
    return gb.Softmax([x], axis=int(axis))


@converter.generic_converter(1)
def softplus(gb, x, beta):
    assert beta == 1.0, 'Softplus with beta != 1.0 is not supported yet.'
    return gb.Softplus([x])


def get_mapping():
    mapping = {
        F.clipped_relu: clipped_relu,
        F.elu: elu,
        F.hard_sigmoid: converter.unary('HardSigmoid'),
        F.leaky_relu: leaky_relu,
        F.log_softmax: log_softmax,
        F.prelu: converter.binary('PRelu'),
        F.relu: converter.unary('Relu'),
        F.sigmoid: converter.unary('Sigmoid'),
        F.softmax: softmax,
        F.softplus: softplus,
        F.tanh: converter.unary('Tanh'),
    }
    return mapping
