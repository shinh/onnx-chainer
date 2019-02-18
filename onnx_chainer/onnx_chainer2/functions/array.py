from __future__ import print_function


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


def expand_dims(gb, x, axis):
    if axis < 0:
        axis += len(gb.shape(x)) + 1
    return gb.Unsqueeze([x], axes=[axis])


def get_item(gb, x, slices):
    if isinstance(slices, list):
        if all([isinstance(s, int) for s in slices]):
            slices = slices,
        slices = tuple(slices)
    elif not isinstance(slices, tuple):
        slices = slices,

    axes, starts, ends = [], [], []
    squeeze_idxs, unsqueeze_idxs = [], []
    skipped = 0  # when set ellipsis, need to skip index rolling
    int_max = 2 ** 31 - 1

    for i, idx in enumerate(slices):
        # axis means the index of input x, adjust None and Ellipsis counts
        axis = i - len(unsqueeze_idxs) + skipped
        if isinstance(idx, slice):
            if idx.step is not None and idx.step != 1:
                raise ValueError(
                    'GetItem with {}step slicing is not supported in ONNX '
                    'Slice operator'.format(idx.step))
            axes.append(axis)
            starts.append(0 if idx.start is None else idx.start)
            ends.append(int_max if idx.stop is None else idx.stop)
        elif isinstance(idx, int):
            axes.append(axis)
            starts.append(idx)
            ends.append(idx+1)
            squeeze_idxs.append(axis)
        elif isinstance(idx, np.ndarray) and idx.ndim == 0:
            scalar_idx = np.asscalar(idx)
            axes.append(axis)
            starts.append(scalar_idx)
            ends.append(scalar_idx+1)
            squeeze_idxs.append(axis)
        elif idx is None:
            unsqueeze_idxs.append(i - len(squeeze_idxs) + skipped)
        elif idx is Ellipsis:
            # TODO(hamaji): Implement ellipsis for onnx_chainer2.
            raise ValueError(
                'GetItem with ellipsis is not supported by onnx_chainer2 yet.')
            # calculate rest slice number except None, GetItem does not allow
            # multiple Ellipsis, so ignore latter Ellipsis count
            rest_slice_len = len(
                [idx_ for idx_ in slices[i+1:] if idx_ is not None])
            assert skipped == 0
            skipped = len(x.shape) - axis - rest_slice_len - 1
        else:
            # not support advanced index like `array[[0,1], [0, 1]]`
            raise ValueError(
                'GetItem with type {} cannot handle in ONNX Slice, so that '
                'ONNX-Chainer does not accept the type'.format(type(idx)))

    result = gb.Slice([x], axes=axes, starts=starts, ends=ends)
    if squeeze_idxs:
        result = squeeze(gb, result.output[0], squeeze_idxs)
    if unsqueeze_idxs:
        # TODO(hamaji): Implement expand_dims and use it.
        result = gb.Unsqueeze([result.output[0]], axes=unsqueeze_idxs)
    return result


def pad(gb, x, pad_width, mode, **keywords):
    # TODO(hamaji): Implement this.
    assert False


def reshape(gb, x, shape):
    return gb.Reshape([x, shape])


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


def transpose(gb, x, axes):
    kwargs = {}
    if axes is not None:
        kwargs['perm'] = axes
    return gb.Transpose([x], **kwargs)


def get_mapping():
    mapping = {
        F.cast: cast,
        F.concat: concat,
        F.copy: copy,
        F.depth2space: depth2space,
        F.expand_dims: expand_dims,
        F.get_item: get_item,
        F.pad: pad,
        F.space2depth: space2depth,
        F.split_axis: split_axis,
        F.squeeze: squeeze,
        F.tile: tile,
        F.transpose: transpose,
    }
    mapping = {k: converter.generic(f, 1) for k, f in mapping.items()}
    mapping[F.reshape] = converter.generic(reshape, 2)
    return mapping
