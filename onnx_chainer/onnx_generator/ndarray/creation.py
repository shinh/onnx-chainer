from __future__ import print_function

import numpy as np
from onnx_chainer.onnx_generator import array_modules
from onnx_chainer.onnx_generator import converter


def zeros(gb, shape, dtype=None, order='C'):
    # TODO(hamaji): Use ConstantOfShape once onnxruntime supports it.
    a = np.zeros(shape, dtype, order)
    return gb.const(a, name='zeros')


def ones(gb, shape, dtype=None, order='C'):
    # TODO(hamaji): Use ConstantOfShape once onnxruntime supports it.
    a = np.ones(shape, dtype, order)
    return gb.const(a, name='ones')


def get_mapping():
    mapping = {}
    for xp in array_modules.get_array_modules():
        for name, converter_fn in {
                'zeros': converter.builtin(zeros, 0),
                'ones': converter.builtin(ones, 0),
        }.items():
            real_fn = getattr(xp, name)
            mapping[real_fn] = converter_fn
    return mapping
