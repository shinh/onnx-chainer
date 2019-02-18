import chainer.functions as F
import numpy as np
from onnx_chainer.onnx_generator import converter


def _pair(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v, v]


def convolution_2d(gb, x, w, b, stride, pad, cover_all, dilate=1, groups=1):
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
    kernel_shape = gb.shape(w)[2:]
    # TODO(hamaji): Handle dilation.
    return gb.Conv(inputs,
                   kernel_shape=kernel_shape,
                   pads=_pair(pad) * 2,
                   strides=_pair(stride),
                   group=groups)


def linear(gb, x, w, b, n_batch_axes):
    assert n_batch_axes == 1, \
        'n_batch_axes != 1 for linear is not supported yet'

    # TODO(hamaji): Support graphs with dynamic shapes.
    x_shape = gb.shape(x)
    if len(x_shape) != 2:
        shape = gb.const(np.array([x_shape[0], np.prod(x_shape[1:])]))
        x = gb.Reshape([x, shape.output[0]]).output[0]

    if b is None:
        t = gb.Transpose([w], perm=[1, 0])
        return gb.MatMul([x, t.output[0]])
    else:
        return gb.Gemm([x, w, b],
                       alpha=1.0,
                       beta=1.0,
                       transA=0,
                       transB=1)


def get_mapping():
    mapping = {
        F.convolution_2d: converter.generic(convolution_2d, 3),
        F.linear: converter.generic(linear, 3),
    }
    return mapping
