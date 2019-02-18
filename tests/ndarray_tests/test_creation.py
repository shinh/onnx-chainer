import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np
import onnx

import onnx_chainer
from onnx_chainer.testing import input_generator
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'name': 'zeros'},
    {'name': 'ones'},
)
class TestCreation(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, shape):
                super(Model, self).__init__()
                self.ops = ops
                self.shape = shape

            def __call__(self):
                return chainer.Variable(self.ops(self.shape))

        ops = getattr(np, self.name)
        self.model = Model(ops, (2, 3, 5))
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                onnx_chainer.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, (), self.fn, opset_version=opset_version,
                onnx_chainer2=True)
