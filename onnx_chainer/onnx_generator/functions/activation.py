from __future__ import print_function

import chainer.functions as F
from onnx_chainer.onnx_generator import converter


def clipped_relu(gb, x, z):
    return gb.Clip([x], min=0.0, max=float(z))


def elu(gb, x, alpha):
    return gb.Elu([x], alpha=float(alpha))


def leaky_relu(gb, x, slope):
    return gb.LeakyRelu([x], alpha=float(slope))


def log_softmax(gb, x, axis):
    assert axis == 1, 'LogSoftmax with axis != 1 is not supported yet.'
    return gb.LogSoftmax([x], axis=int(axis))


def softmax(gb, x, axis):
    assert axis == 1, 'Softmax with axis != 1 is not supported yet.'
    return gb.Softmax([x], axis=int(axis))


def softplus(gb, x, beta):
    assert beta == 1.0, 'Softplus with beta != 1.0 is not supported yet.'
    return gb.Softplus([x])


def get_mapping():
    mapping = {
        F.clipped_relu: converter.generic(clipped_relu, 1),
        F.elu: converter.generic(elu, 1),
        F.hard_sigmoid: converter.unary('HardSigmoid'),
        F.leaky_relu: converter.generic(leaky_relu, 1),
        F.log_softmax: converter.generic(log_softmax, 1),
        F.prelu: converter.binary('PRelu'),
        F.relu: converter.unary('Relu'),
        F.sigmoid: converter.unary('Sigmoid'),
        F.softmax: converter.generic(softmax, 1),
        F.softplus: converter.generic(softplus, 1),
        F.tanh: converter.unary('Tanh'),
    }
    return mapping
