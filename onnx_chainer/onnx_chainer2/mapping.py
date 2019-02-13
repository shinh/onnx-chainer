from __future__ import print_function

import inspect

from onnx_chainer.onnx_chainer2.functions import activation


def get_converter():
    mapping = {}
    mapping.update(activation.get_mapping())

    def convert(gb, real_fn, me, args, kwargs):
        converter = mapping[real_fn]
        sig = inspect.signature(real_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return converter(gb, *bound.args, **bound.kwargs)

    return convert
