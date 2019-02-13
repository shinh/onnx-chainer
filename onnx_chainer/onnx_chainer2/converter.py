from __future__ import print_function


def generic(func, num_inputs):
    def converter(gb, *args, **kwargs):
        new_args = list(args)
        for i, arg in list(enumerate(args))[:num_inputs]:
            if arg is not None:
                new_args[i] = gb.get_value_name(arg)
        return func(gb, *new_args, **kwargs)
    return converter


def unary(onnx_op):
    def fn(gb, x):
        return getattr(gb, onnx_op)([x])
    return generic(fn, 1)


def binary(onnx_op):
    def fn(gb, x, y):
        return getattr(gb, onnx_op)([x, y])
    return generic(fn, 2)
