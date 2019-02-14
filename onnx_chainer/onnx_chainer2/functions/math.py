from __future__ import print_function

import chainer
import chainer.functions as F
import numpy as np
import onnx
from onnx_chainer.onnx_chainer2 import converter


def average(gb, x, axis, weight, keepdims):
    if weight is not None:
        raise RuntimeError('F.average with weight!=None is not supported yet')
    kwargs = {'keepdims': keepdims}
    if axis is not None:
        if not isinstance(axis, (list, tuple)):
            axis = [axis]
        kwargs['axes'] = axis
    return gb.ReduceMean([x], **kwargs)


def matmul(gb, a, b, transa, transb):
    if transa:
        a = gb.Transpose([a]).output[0]
    if transb:
        b = gb.Transpose([b]).output[0]
    return gb.MatMul([a, b])


def get_mapping():
    mapping = {
        F.average: converter.generic(average, 1),
        F.matmul: converter.generic(matmul, 2),
        F.maximum: converter.binary('Max'),
        F.minimum: converter.binary('Min'),
        chainer.Variable.__add__: converter.binary('Add'),
        chainer.Variable.__truediv__: converter.binary('Div'),
        chainer.Variable.__mul__: converter.binary('Mul'),
        chainer.Variable.__neg__: converter.unary('Neg'),
        chainer.Variable.__sub__: converter.binary('Sub'),
    }
    return mapping
