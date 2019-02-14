from __future__ import print_function

import chainer
import chainer.functions as F
import numpy as np
import onnx
from onnx_chainer.onnx_chainer2 import converter


def get_mapping():
    mapping = {
        chainer.Variable.__neg__: converter.unary('Neg'),
    }
    return mapping
