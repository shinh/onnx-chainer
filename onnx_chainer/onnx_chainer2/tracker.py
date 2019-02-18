import contextlib
import types

import numpy as np

import chainer

from onnx_chainer.onnx_chainer2 import array_modules

_tracker = None
_wrap_array_types = {}


def _wrap_array(array):
    if type(array) in _wrap_array_types:
        array = _wrap_array_types[type(array)](array)
    return array


def _real_array(array):
    if hasattr(array, '_real_array'):
        array = array._real_array()
    return array


def _real_array_args(args, kwargs):
    args = [_real_array(a) for a in args]
    kwargs = {k: _real_array(a) for k, a in kwargs.items()}
    return args, kwargs


class WrapNdArray(object):
    def __init__(self, array):
        assert not hasattr(array, '_real_array')
        self.__real_array = array

    def __getattr__(self, name):
        real_attr = getattr(self.__real_array, name)
        if callable(real_attr) and name != '__array_prepare__':
            def fn(*args, **kwargs):
                args, kwargs = _real_array_args(args, kwargs)
                with _tracker.off_the_record():
                    result = real_attr(*args, **kwargs)
                _tracker.add_record(
                    self.__real_array, name, args, kwargs, result)
                return _tracker.wrap_array(result)
            return fn
        else:
            # TODO(hamaji): Handle properties.
            return real_attr

    def __abs__(self, *args):
        return self.__getattr__('__abs__')(*args)

    def __add__(self, *args):
        return self.__getattr__('__add__')(*args)

    def __and__(self, *args):
        return self.__getattr__('__and__')(*args)

    def __concat__(self, *args):
        return self.__getattr__('__concat__')(*args)

    def __contains__(self, *args):
        return self.__getattr__('__contains__')(*args)

    def __delitem__(self, *args):
        return self.__getattr__('__delitem__')(*args)

    def __eq__(self, *args):
        return self.__getattr__('__eq__')(*args)

    def __floordiv__(self, *args):
        return self.__getattr__('__floordiv__')(*args)

    def __ge__(self, *args):
        return self.__getattr__('__ge__')(*args)

    def __getitem__(self, *args):
        return self.__getattr__('__getitem__')(*args)

    def __gt__(self, *args):
        return self.__getattr__('__gt__')(*args)

    def __index__(self, *args):
        return self.__getattr__('__index__')(*args)

    def __inv__(self, *args):
        return self.__getattr__('__inv__')(*args)

    def __invert__(self, *args):
        return self.__getattr__('__invert__')(*args)

    def __le__(self, *args):
        return self.__getattr__('__le__')(*args)

    def __lshift__(self, *args):
        return self.__getattr__('__lshift__')(*args)

    def __lt__(self, *args):
        return self.__getattr__('__lt__')(*args)

    def __matmul__(self, *args):
        return self.__getattr__('__matmul__')(*args)

    def __mod__(self, *args):
        return self.__getattr__('__mod__')(*args)

    def __mul__(self, *args):
        return self.__getattr__('__mul__')(*args)

    def __ne__(self, *args):
        return self.__getattr__('__ne__')(*args)

    def __neg__(self, *args):
        return self.__getattr__('__neg__')(*args)

    def __not__(self, *args):
        return self.__getattr__('__not__')(*args)

    def __or__(self, *args):
        return self.__getattr__('__or__')(*args)

    def __pos__(self, *args):
        return self.__getattr__('__pos__')(*args)

    def __pow__(self, *args):
        return self.__getattr__('__pow__')(*args)

    def __rshift__(self, *args):
        return self.__getattr__('__rshift__')(*args)

    def __setitem__(self, *args):
        return self.__getattr__('__setitem__')(*args)

    def __sub__(self, *args):
        return self.__getattr__('__sub__')(*args)

    def __truediv__(self, *args):
        return self.__getattr__('__truediv__')(*args)

    def __xor__(self, *args):
        return self.__getattr__('__xor__')(*args)

    def __radd__(self, *args):
        return self.__getattr__('__radd__')(*args)

    def __rsub__(self, *args):
        return self.__getattr__('__rsub__')(*args)

    def __rmul__(self, *args):
        return self.__getattr__('__rmul__')(*args)

    def __rdiv__(self, *args):
        return self.__getattr__('__rdiv__')(*args)

    def __rtruediv__(self, *args):
        return self.__getattr__('__rtruediv__')(*args)

    def __rfloordiv__(self, *args):
        return self.__getattr__('__rfloordiv__')(*args)

    def __rmod__(self, *args):
        return self.__getattr__('__rmod__')(*args)

    def __rdivmod__(self, *args):
        return self.__getattr__('__rdivmod__')(*args)

    def __rpow__(self, *args):
        return self.__getattr__('__rpow__')(*args)

    def __rlshift__(self, *args):
        return self.__getattr__('__rlshift__')(*args)

    def __rrshift__(self, *args):
        return self.__getattr__('__rrshift__')(*args)

    def __rand__(self, *args):
        return self.__getattr__('__rand__')(*args)

    def __rxor__(self, *args):
        return self.__getattr__('__rxor__')(*args)

    def __ror__(self, *args):
        return self.__getattr__('__ror__')(*args)

    def _real_array(self):
        return self.__real_array


class WrapNumPyArray(WrapNdArray):
    pass

_wrap_array_types[np.ndarray] = WrapNumPyArray

for xp in array_modules.get_array_modules():
    if xp.__name__ == 'cupy':
        class WrapCupyArray(WrapNdArray):
            pass

        _wrap_array_types[xp.ndarray] = WrapCupyArray

    elif xp.__name__ == 'chainerx':
        class WrapChainerXArray(WrapNdArray):
            pass

        _wrap_array_types[xp.ndarray] = WrapChainerXArray


class WrapChainerVariable(WrapNdArray):
    pass

_wrap_array_types[chainer.Variable] = WrapChainerVariable


def create_wrap_func(module, name, real):
    def fn(*args, **kwargs):
        args, kwargs = _real_array_args(args, kwargs)
        with _tracker.off_the_record():
            result = real(*args, **kwargs)
        _tracker.add_record(module, name, args, kwargs, result)
        return _tracker.wrap_array(result)
    return fn


def wrap_module(module, predefined_funcs=None, recursive=False):
    if predefined_funcs is None:
        predefined_funcs = {}

    replaced = []
    sub_modules = []
    for name in dir(module):
        if name == 'ndarray':
            continue
        real = getattr(module, name)
        wrap = real
        if name in predefined_funcs:
            wrap = predefined_funcs[name]
        elif isinstance(real, type):
            continue
        elif isinstance(real, types.ModuleType):
            if recursive and module.__name__ in real.__name__:
                sub_modules.append(real)
            continue
        elif callable(real):
            wrap = create_wrap_func(module, name, real)
        replaced.append((name, real, wrap))

    for name, _, wrap in replaced:
        _tracker.wrap_attribute(module, name, wrap)

    for name, real, wrap in replaced:
        _tracker.real2wrap[id(real)] = wrap

    for sub_module in sub_modules:
        wrap_module(sub_module, recursive=True)


class NdArrayLike(tuple):
    def __call__(self, *args, **kwargs):
        return self[0](*args, **kwargs)


class Tracker(object):

    def __init__(self):
        array_types = chainer.get_array_types() + (WrapNumPyArray,)
        chainer_predefined_funcs = {
            'is_arrays_compatible': lambda a: True,
            'get_array_types': lambda: array_types,
        }

        self._wrapped_attributes = []

        global _tracker
        _tracker = self
        self.real2wrap = {}

        for xp in array_modules.get_array_modules():
            wrap_module(xp)
        wrap_module(chainer, chainer_predefined_funcs)
        wrap_module(chainer.functions, recursive=True)

        for xp in array_modules.get_array_modules():
            self.wrap_attribute(
                xp, 'ndarray',
                NdArrayLike([xp.ndarray, _wrap_array_types[xp.ndarray]]))
        self.wrap_attribute(
            chainer, 'Variable',
            NdArrayLike([chainer.Variable, WrapChainerVariable]))

        self._recorded_calls = []
        self._off_the_record_count = 0

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for receiver, name, _, real in reversed(self._wrapped_attributes):
            setattr(receiver, name, real)

    def add_record(self, receiver, name, args, kwargs, result):
        if not self._off_the_record_count:
            self._recorded_calls.append((receiver, name, args, kwargs, result))

    def get_records(self):
        return self._recorded_calls

    @contextlib.contextmanager
    def off_the_record(self):
        self._off_the_record_count += 1
        yield
        self._off_the_record_count -= 1

    def wrap_array(self, array):
        if self._off_the_record_count:
            return array
        return _wrap_array(array)

    def get_wrap(self, real):
        if isinstance(real, (chainer.Sequential, list, tuple)):
            wrap = []
            changed = False
            for r in real:
                w = self.get_wrap(r)
                if w is None:
                    wrap.append(r)
                else:
                    changed = True
                    wrap.append(w)

            if not changed:
                return None
            return type(real)(wrap)

        if not callable(real):
            return None
        return self.real2wrap.get(id(real))

    def wrap_model(self, model):
        for child in model.children():
            self.wrap_model(child)

        for name in dir(model):
            # Do not access the deprecated attribute to avoid warnings.
            if name == '_device_id':
                continue

            real = getattr(model, name)
            wrap = self.get_wrap(real)
            if wrap is not None:
                self.wrap_attribute(model, name, wrap)

    def wrap_attribute(self, receiver, name, wrap):
        real = getattr(receiver, name)
        self._wrapped_attributes.append((receiver, name, wrap, real))
        setattr(receiver, name, wrap)
