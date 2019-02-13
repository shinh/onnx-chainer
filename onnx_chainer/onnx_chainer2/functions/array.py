from __future__ import print_function


import chainer
import chainer.functions as F
import numpy as np
import onnx
from onnx_chainer.onnx_chainer2 import converter


def cast(gb, x, typ):
    typ = typ if isinstance(typ, np.dtype) else np.dtype(typ)
    return gb.Cast([x], to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[typ])


def concat(gb, x, axis):
    return gb.Concat([x], axis=axis)


def copy(gb, x, dst):
    return gb.Identity([x])


def depth2space(gb, x, r):
    return gb.DepthToSpace([x], blocksize=r)


def get_item(gb, x, slices):
    # TODO(hamaji): Implement this.
    assert False


def pad(gb, x, pad_width, mode, **keywords):
    # TODO(hamaji): Implement this.
    assert False


def reshape(gb, x, shape):
    # TODO(hamaji): Implement this.
    assert False
    return gb.Rehsape([x])


def space2depth(gb, x, r):
    return gb.SpaceToDepth([x], blocksize=r)


def split_axis(gb, x, indices_or_sections, axis, force_tuple):
    # TODO(hamaji): Implement this.
    assert False


def squeeze(gb, x, axis):
    kwargs = {}
    if isinstance(axis, (list, tuple)):
        kwargs['axes'] = axis
    elif axis is not None:
        kwargs['axes'] = [int(axis)]
    return gb.Squeeze([x], **kwargs)


def tile(gb, x, reps):
    assert False


def get_mapping():
    mapping = {
        F.cast: cast,
        F.concat: concat,
        F.copy: copy,
        F.depth2space: depth2space,
        F.get_item: get_item,
        F.pad: pad,
        F.reshape: reshape,
        F.space2depth: space2depth,
        F.split_axis: split_axis,
        F.squeeze: squeeze,
        F.tile: tile,
    }
    return {k: converter.generic(f, 1) for k, f in mapping.items()}
