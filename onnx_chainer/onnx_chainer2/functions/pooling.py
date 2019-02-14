import chainer.functions as F
from onnx_chainer.onnx_chainer2 import converter


def _spatial(v, ndim):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v] * ndim


def average_pooling(ndim):
    def fn(gb, x, ksize, stride, pad, pad_value=0):
        if pad_value != 0:
            raise RuntimeError(
                'Average pooling with pad_value!=0 is not supported yet.')
        nd = len(gb.shape(x)) - 2 if ndim is None else ndim
        return gb.AveragePool([x],
                              kernel_shape=_spatial(ksize, nd),
                              pads=_spatial(pad, nd) * 2,
                              strides=_spatial(stride, nd),
                              count_include_pad=1)
    return fn


def max_pooling(ndim):
    def fn(gb, x, ksize, stride, pad, cover_all, return_indices):
        if return_indices:
            raise RuntimeError(
                'Max pooling with return_indices=True is not supported yet')
        nd = len(gb.shape(x)) - 2 if ndim is None else ndim
        kernel_shape = _spatial(ksize, nd)
        pads = _spatial(pad, nd)
        strides = _spatial(stride, nd)

        if cover_all:
            # Supports cover_all by setting extra padding
            for p, s, k in zip(pads, strides, kernel_shape):
                # Raise exception because a virtual pad for cover_all must be
                # smaller than ksize in the current ONNX
                if k <= p + s - 1:
                    raise RuntimeError(
                        'Could not correctly export in the current setting'
                        ' (ksize={} pad={} stride={}). Please set pad or '
                        'stride to lower value.'.format(k, p, s))
            pads.extend([p + s - 1 for p, s in zip(pads, strides)])
        else:
            pads = pads * 2

        return gb.MaxPool([x],
                          kernel_shape=kernel_shape,
                          pads=pads,
                          strides=strides)
    return fn


def get_mapping():
    # TODO(hamaji): Test 1D and 3D poolings.
    mapping = {
        F.average_pooling_1d: average_pooling(1),
        F.average_pooling_2d: average_pooling(2),
        F.average_pooling_3d: average_pooling(3),
        F.average_pooling_nd: average_pooling(None),
        F.max_pooling_1d: max_pooling(1),
        F.max_pooling_2d: max_pooling(2),
        F.max_pooling_3d: max_pooling(3),
        F.max_pooling_nd: max_pooling(None),
    }
    mapping = {k: converter.generic(f, 1) for k, f in mapping.items()}
    return mapping
