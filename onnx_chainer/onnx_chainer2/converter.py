from __future__ import print_function


class FunctionConverter(object):
    def __init__(self, fn):
        self.sig = inspect.signature(fn)

    def convert(self, gb, args, kwargs):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args = [Value(a) for a in bound.args]
        kwargs = {k: Value(a) for k, a in bound.kwargs.items()}
        return self.convert_impl(gb, *args, **kwargs)

    def convert_impl(self, gb, *args, **kwargs):
        raise NotImplementedError('convert_impl must be implemented')


def generic_converter(num_inputs):
    def deco(func):
        def converter(gb, *args):
            new_args = list(args)
            for i in range(num_inputs):
                new_args[i] = gb.get_value_name(new_args[i])
            return func(gb, *new_args)
        return converter
    return deco


def unary(onnx_op):
    @generic_converter(1)
    def fn(gb, x):
        return getattr(gb, onnx_op)([x])
    return fn
