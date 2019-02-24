from __future__ import print_function

import functools
import inspect


class Converter(object):
    def __init__(self, converter_fn, num_tensor_inputs, is_builtin):
        self.converter_fn = converter_fn
        self.num_tensor_inputs = num_tensor_inputs
        self.is_builtin = is_builtin

    def convert(self, gb, real_fn, args, kwargs):
        if self.is_builtin:
            sig_fn = functools.partial(self.converter_fn, gb)
        else:
            sig_fn = real_fn
        sig = inspect.signature(sig_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        new_args = []
        for i, arg in enumerate(bound.args):
            if i < self.num_tensor_inputs:
                if arg is not None:
                    arg = gb.get_value_name(arg)
            else:
                if hasattr(arg, 'value'):
                    arg = arg.value
            new_args.append(arg)
        new_kwargs = {}
        for k, arg in bound.kwargs.items():
            if hasattr(arg, 'value'):
                arg = arg.value
            new_kwargs[k] = arg
        return self.converter_fn(gb, *new_args, **new_kwargs)


def generic(converter_fn, num_tensor_inputs):
    return Converter(converter_fn, num_tensor_inputs, is_builtin=False)


def builtin(converter_fn, num_tensor_inputs):
    return Converter(converter_fn, num_tensor_inputs, is_builtin=True)


def unary(onnx_op):
    def fn(gb, x):
        return getattr(gb, onnx_op)([x])
    return generic(fn, 1)


def binary(onnx_op):
    def fn(gb, x, y):
        return getattr(gb, onnx_op)([x, y])
    return generic(fn, 2)
