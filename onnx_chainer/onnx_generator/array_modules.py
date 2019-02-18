import numpy as np

__array_modules = [np]

try:
    import cupy
    __array_modules.append(cupy)
except ImportError:
    pass

try:
    import chainerx
    __array_modules.append(chainerx)
except ImportError:
    pass


def get_array_modules():
    return __array_modules
