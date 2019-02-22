import contextlib
import types

import numpy as np

import chainer

from onnx_chainer.onnx_generator import array_modules

_tracker = None
_wrap_types = {}


def _wrap_value(array):
    if type(array) in _wrap_types:
        array = _wrap_types[type(array)](array)
    return array


def _real_value(array):
    if hasattr(array, '_real_value'):
        array = array._real_value()
    return array


def _real_value_args(args, kwargs):
    args = [_real_value(a) for a in args]
    kwargs = {k: _real_value(a) for k, a in kwargs.items()}
    return args, kwargs


def _real_array(array):
    real = _real_value(array)
    if isinstance(real, chainer.Variable):
        real = real.array
    return real


class ValueInfo(object):
    def __init__(self, vid, op_name, typ, shape, dtype, value):
        self.op_name = op_name
        self.vid = vid
        self.typ = typ
        self.shape = shape
        self.dtype = dtype
        self.value = value

    def __repr__(self):
        toks = {
            'op_name': self.op_name,
            'vid': self.vid,
            'shape': self.shape,
            'dtype': self.dtype,
        }
        if self.value is not None:
            toks['value'] = self.value
        strs = ['%s=%s' % (k, str(v)) for k, v in toks.items()]
        return 'ValueInfo(%s)' % ' '.join(strs)


_value_infos = {}


def _value_info(value, op_name, is_input=True):
    assert not isinstance(value, ValueInfo)
    vid = id(value)
    real = _real_array(value)
    print('vid=%s name=%s %s' % (vid, op_name, id(real)))
    shape = getattr(real, 'shape', None)
    dtype = getattr(real, 'dtype', None)

    if vid in _value_infos:
        vi = _value_infos[vid]
        assert vi.shape == shape, '%s vs %s' % (vi, real)
        assert vi.dtype == dtype, '%s vs %s' % (vi, real)
        return vi

    vi = ValueInfo(vid, op_name, type(real), shape, dtype,
                   real if is_input else None)
    _value_infos[vid] = vi
    return vi


def _value_info_list(values, op_name, is_input=True):
    return [_value_info(v, op_name, is_input=is_input) for v in values]


def _value_info_dict(values, op_name, is_input=True):
    return {k: _value_info(v, op_name, is_input=is_input)
            for k, v in values.items()}


def _value_info_result(result, op_name):
    if (isinstance(result, (list, tuple)) and
        len(result) > 0 and
        isinstance(result[0],
                   [chainer.Variable] + list(chainer.get_array_types()))):
        return _value_info_list(result, op_name, is_input=True)
    if (isinstance(result, dict) and
        len(result) > 0 and
        isinstance(list(result.values())[0],
                   [chainer.Variable] + list(chainer.get_array_types()))):
        return _value_info_dict(result, op_name, is_input=True)
    return _value_info(result, op_name, is_input=True)


class WrapValue(object):
    def __init__(self, array):
        assert not hasattr(array, '_real_value')
        self.__real_value = array

    def __getattr__(self, name):
        real_attr = getattr(self.__real_value, name)
        if (callable(real_attr) and
            name not in ('__array_prepare__', '__repr__', '__str__')):
            def fn(*wrap_args, **wrap_kwargs):
                args, kwargs = _real_value_args(wrap_args, wrap_kwargs)
                with _tracker.off_the_record():
                    result = real_attr(*args, **kwargs)
                wrap_result = _tracker.wrap_value(result)
                _tracker.add_record(self, name,
                                    wrap_args, wrap_kwargs, wrap_result)
                return wrap_result
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

    def _real_value(self):
        return self.__real_value


class WrapNumPyArray(WrapValue):
    pass


_wrap_types[np.ndarray] = WrapNumPyArray

for xp in array_modules.get_array_modules():
    if xp.__name__ == 'cupy':
        class WrapCupyArray(WrapValue):
            pass

        _wrap_types[xp.ndarray] = WrapCupyArray

    elif xp.__name__ == 'chainerx':
        class WrapChainerXArray(WrapValue):
            pass

        _wrap_types[xp.ndarray] = WrapChainerXArray


class WrapChainerVariable(WrapValue):
    pass


_wrap_types[chainer.Variable] = WrapChainerVariable


def create_wrap_func(module, name, real):
    def fn(*wrap_args, **wrap_kwargs):
        args, kwargs = _real_value_args(wrap_args, wrap_kwargs)
        with _tracker.off_the_record():
            result = real(*args, **kwargs)
        wrap_result = _tracker.wrap_value(result)
        _tracker.add_record(module, name, wrap_args, wrap_kwargs, wrap_result)
        return wrap_result
    return fn


def wrap_module(module, predefined_funcs=None, recursive=False):
    if predefined_funcs is None:
        predefined_funcs = {}

    replaced = []
    sub_modules = []
    for name in dir(module):
        if name in ['ndarray', 'array2string']:
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


class WrappedType(tuple):
    """A hack to keep `isinstance(x, np.ndarray)` working."""
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
                WrappedType([xp.ndarray, _wrap_types[xp.ndarray]]))
        self.wrap_attribute(
            chainer, 'Variable',
            WrappedType([chainer.Variable, WrapChainerVariable]))

        self._recorded_calls = []
        self._off_the_record_count = 0

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for receiver, name, _, real in reversed(self._wrapped_attributes):
            setattr(receiver, name, real)

    def add_record(self, receiver, name, args, kwargs, result):
        if self._off_the_record_count:
            return

        func = getattr(receiver, name)
        receiver = _value_info(receiver, name)
        args = _value_info_list(args, name)
        kwargs = _value_info_dict(kwargs, name)
        result = _value_info_result(result, name)
        # When the `receiver` is a bound method.
        if hasattr(func, '__func__'):
            args.insert(0, _value_info(func.__self__, name))
            func = func.__func__
        elif hasattr(func, '__self__'):
            assert hasattr(func, '__name__')
            args.insert(0, _value_info(func.__self__))
            func = getattr(type(func.__self__), func.__name__)
        self._recorded_calls.append(
            (name, func, receiver, args, kwargs, result))

    def get_records(self):
        return self._recorded_calls

    @contextlib.contextmanager
    def off_the_record(self):
        self._off_the_record_count += 1
        yield
        self._off_the_record_count -= 1

    def wrap_value(self, array):
        if self._off_the_record_count:
            return array
        return _wrap_value(array)

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
