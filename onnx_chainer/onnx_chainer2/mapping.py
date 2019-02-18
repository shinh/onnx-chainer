from __future__ import print_function

import inspect

from onnx_chainer.onnx_chainer2.functions import activation
from onnx_chainer.onnx_chainer2.functions import array
from onnx_chainer.onnx_chainer2.functions import connection
from onnx_chainer.onnx_chainer2.functions import math
from onnx_chainer.onnx_chainer2.functions import normalization
from onnx_chainer.onnx_chainer2.functions import pooling


def _merge_mapping(mapping, new_mapping):
    for key_fn, converter_fn in new_mapping.items():
        # Use id of `key_fn` since some keys are not hashable (e.g.,
        # `chainer.Variable`).
        if isinstance(key_fn, int):
            key_id = key_fn
        else:
            assert callable(key_fn)
            key_id = id(key_fn)
        assert key_id not in mapping, 'Duplicated mapping: %s' % key_fn
        mapping[key_id] = converter_fn


def get_converter():
    mapping = {}
    _merge_mapping(mapping, activation.get_mapping())
    _merge_mapping(mapping, array.get_mapping())
    _merge_mapping(mapping, connection.get_mapping())
    _merge_mapping(mapping, math.get_mapping())
    _merge_mapping(mapping, normalization.get_mapping())
    _merge_mapping(mapping, pooling.get_mapping())

    def convert(gb, real_fn, me, args, kwargs):
        real_id = id(real_fn)
        if real_id not in mapping:
            raise RuntimeError('%s is not supported yet' % real_fn)
        converter = mapping[real_id]
        return converter.convert(gb, real_fn, args, kwargs)

    return convert
