import numpy as np

import chainer

__array_modules = [np]

try:
    import cupy
    __array_modules.append(cupy)
except:
    pass

try:
    import chainerx
    __array_modules.append(chainerx)
except:
    pass


def get_array_modules():
    return __array_modules
