import chainer.functions as F
from onnx_chainer.onnx_generator import converter


def batch_normalization(gb, x, gamma, beta,
                        eps, running_mean, running_var, decay, axis):
    # TODO(hamaji): Add test for this path and fix.
    raise RuntimeError(
        'BatchNormalization for training is not implemented yet')


def fixed_batch_normalization(gb, x, gamma, beta, mean, var, eps, axis):
    assert axis is None, 'BatchNormalization with axis is not supported yet'
    return gb.BatchNormalization([x, gamma, beta, mean, var],
                                 epsilon=eps)


def get_mapping():
    mapping = {
        F.batch_normalization: converter.generic(batch_normalization, 3),
        F.fixed_batch_normalization: converter.generic(
            fixed_batch_normalization, 5),
    }
    return mapping
