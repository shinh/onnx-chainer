import contextlib

import numpy as np

import chainer


_tracker = None
__wrap_array_types = {}


def _wrap_array(array):
    if type(array) in __wrap_array_types:
        array = __wrap_array_types[type(array)](array)
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

    def __getitem__(self, *args):
        return self.__getattr__('__getitem__')(*args)

    def __setitem__(self, *args):
        return self.__getattr__('__setitem__')(*args)

    def __add__(self, *args):
        return self.__getattr__('__add__')(*args)

    def _real_array(self):
        return self.__real_array


class WrapNumPyArray(WrapNdArray):
    pass


class WrapChainerVariable(WrapNdArray):
    pass


__wrap_array_types[np.ndarray] = WrapNumPyArray
__wrap_array_types[chainer.Variable] = WrapChainerVariable


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
        # TODO(hamaji): Find a better way to get this type.
        module_type = type(chainer)
        if name in predefined_funcs:
            wrap = predefined_funcs[name]
        elif isinstance(real, type):
            continue
        elif isinstance(real, module_type):
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

        wrap_module(np)
        wrap_module(chainer, chainer_predefined_funcs)
        wrap_module(chainer.functions, recursive=True)

        # TODO(hamaji): Figure out a better way to handle isinstance.
        self.wrap_attribute(np, 'ndarray',
                            NdArrayLike([np.ndarray, WrapNumPyArray]))

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
