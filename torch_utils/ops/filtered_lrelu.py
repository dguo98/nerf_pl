import os
import numpy as np
import torch

from .. import custom_ops
from .. import misc
from . import upfirdn2d
from . import bias_act

#----------------------------------------------------------------------------

default_impl = 'cuda'
_plugin = None

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='filtered_lrelu_plugin',
            sources=['filtered_lrelu.cpp',
                'filtered_lrelu_float16_wr.cu', 'filtered_lrelu_float32_wr.cu', 'filtered_lrelu_float64_wr.cu',
                'filtered_lrelu_float16_rd.cu', 'filtered_lrelu_float32_rd.cu', 'filtered_lrelu_float64_rd.cu'],
            headers=['filtered_lrelu.h', 'filtered_lrelu.cu'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
    return True

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 3
    size = f.shape[-1]
    if f.ndim == 1:
        rank = 1
    elif f.ndim == 2:
        rank = -1
        misc.assert_shape(f, [size, size])
    else:
        rank = f.shape[-2]
        misc.assert_shape(f, [2, rank, size])
    return size, rank

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

#----------------------------------------------------------------------------

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl=None, variant=0):
    assert isinstance(x, torch.Tensor)
    if impl is None:
        impl = default_impl
    assert impl in ['ref', 'cuda']

    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        fut = (fu.shape[-1] - 1) if fu is not None else 0
        s_zero_ofs = min(fut - _parse_padding(padding)[0], 0) # Offset to x=0 in sign tensor.
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter, s_zero_ofs=s_zero_ofs, variant=variant).apply(x, fu, fd, b, None)

    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)

#----------------------------------------------------------------------------

@misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    fu_size, _fu_rank = _get_filter_size(fu)
    fd_size, _fd_rank = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_size - 1) - (fd_size - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_size - 1) - (fd_size - 1) + (down - 1)) // down

    # Apply bias.
    x = bias_act.bias_act(x=x, b=b)

    # Upsample.
    if fu is None or fu.ndim <= 2:
        x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0,px1,py0,py1], gain=up**2, flip_filter=flip_filter)
    else:
        t = x
        x = None
        for fx, fy in fu.unbind(1):
            y = upfirdn2d.upfirdn2d(x=t, f=fx.unsqueeze(0), up=[up,1], padding=[px0,px1,0,0], gain=up, flip_filter=flip_filter)
            y = upfirdn2d.upfirdn2d(x=y, f=fy.unsqueeze(1), up=[1,up], padding=[0,0,py0,py1], gain=up, flip_filter=flip_filter)
            x = y if x is None else x.add_(y)

    # Bias, leaky ReLU, clamp.
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)

    # Downsample.
    if fd is None or fd.ndim <= 2:
        x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)
    else:
        t = x
        x = None
        for fx, fy in fd.unbind(1):
            y = upfirdn2d.upfirdn2d(x=t, f=fx.unsqueeze(0), down=[down,1], flip_filter=flip_filter)
            y = upfirdn2d.upfirdn2d(x=y, f=fy.unsqueeze(1), down=[1,down], flip_filter=flip_filter)
            x = y if x is None else x.add_(y)

    # Check output shape & dtype.
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x

#----------------------------------------------------------------------------

_filtered_lrelu_cuda_cache = dict()

def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, s_zero_ofs=0, variant=0):
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter, s_zero_ofs, variant)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    # Forward op.
    class FilteredLReluCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b, si): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4

            if fu is None:
                fu = torch.ones([1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 3
            assert 1 <= fd.ndim <= 3
            fu = fu.expand([1] * (3 - fu.ndim) + list(fu.shape))
            fd = fd.expand([1] * (3 - fd.ndim) + list(fd.shape))

            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)
            if si is None:
                si = torch.empty([0])

            y, so = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, py0, px1, py1, gain, slope, clamp, flip_filter, s_zero_ofs, variant)
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            dx  = None # 0
            dfu = None; assert not ctx.needs_input_grad[1]
            dfd = None; assert not ctx.needs_input_grad[2]
            db  = None # 3
            dsi = None; assert not ctx.needs_input_grad[4]

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff, s_zero_ofs=s_zero_ofs, variant=variant).apply(dy, fd, fu, None, si)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda

#----------------------------------------------------------------------------
