import numpy as np

from numba import jit
from functools import reduce

from numba_extensions.utils import meshgrid_float


#######################################################################################################################
# bilateral filter
#######################################################################################################################


@jit('int32(int32)')
def get_window_size(idx):
    return [7, 7, 5, 5, 5][idx]


@jit('float32[:,:](int32, float32)')
def compute_spatial_kernel(mid_point, sigma_s):
    ax = np.arange(-mid_point, mid_point + 1.)

    len_ax = len(ax)
    xx = np.zeros((len_ax, len_ax), dtype=np.int32)
    yy = np.zeros((len_ax, len_ax), dtype=np.int32)
    meshgrid_float(ax, ax, xx, yy)

    spatial_term = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma_s ** 2))

    return spatial_term.astype(np.float32)


def padding(depth, mid_point):
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1, 1), (1, 1)), 'edge')
    pad_depth = np.pad(depth, (mid_point, mid_point), 'edge')

    return depth, pad_depth


def bilateral_filter(depth, discontinuity_map=None, mask=None, window_size=False, sigma_s=4.0, sigma_r=0.5):
    pad_discontinuity_hole_patches = None
    pad_discontinuity_patches = None
    pad_mask_patches = None
    spatial_term = None

    if window_size is False:
        window_size = [7, 7, 5, 5, 5]

    mid_point = window_size // 2
    if discontinuity_map is not None:
        spatial_term = compute_spatial_kernel(mid_point, sigma_s)

    depth, pad_depth = padding(depth, mid_point)

    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1, 1), (1, 1)), 'edge')
        pad_discontinuity_map = np.pad(discontinuity_map, (mid_point, mid_point), 'edge')
        pad_discontinuity_hole = 1 - pad_discontinuity_map
        pad_discontinuity_patches = rolling_window(pad_discontinuity_map, [window_size, window_size], [1, 1])
        pad_discontinuity_hole_patches = rolling_window(pad_discontinuity_hole, [window_size, window_size], [1, 1])

    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1, 1])

    if mask is not None:
        pad_mask = np.pad(mask, (mid_point, mid_point), 'constant')
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1, 1])

    if discontinuity_map is not None:
        if mask is None:
            if pad_discontinuity_hole_patches is None:
                _bilateral_filter_1(output, pad_depth_patches, np.zeros((0, 0), dtype=np.uint8), discontinuity_map,
                                    pad_discontinuity_patches, np.zeros((0, 0, 0, 0), dtype=np.float32), window_size,
                                    np.zeros((0, 0), dtype=np.float32), sigma_r, spatial_term)
            else:
                _bilateral_filter_1(output, pad_depth_patches, np.zeros((0, 0), dtype=np.uint8), discontinuity_map,
                                    pad_discontinuity_patches, pad_discontinuity_hole_patches, window_size,
                                    np.zeros((0, 0), dtype=np.float32), sigma_r, spatial_term)
        else:
            if pad_discontinuity_hole_patches is None:
                _bilateral_filter_1(output, pad_depth_patches, mask, discontinuity_map, pad_discontinuity_patches,
                                    np.zeros((0, 0, 0, 0), dtype=np.float32), window_size, pad_mask_patches, sigma_r,
                                    spatial_term)
            else:
                _bilateral_filter_1(output, pad_depth_patches, mask, discontinuity_map, pad_discontinuity_patches,
                                    pad_discontinuity_hole_patches, window_size, pad_mask_patches, sigma_r,
                                    spatial_term)
    else:
        _bilateral_filter_2(output, pad_depth_patches, discontinuity_map, pad_discontinuity_patches,
                            window_size, sigma_r, spatial_term)

    return output


@jit('void(float32[:,:],float32[:,:,:,:],uint8[:,:],float32[:,:],float32[:,:,:,:],float32[:,:,:,:],int64,float32[:,:],float64,float32[:,:])')
def _bilateral_filter_1(output, pad_depth_patches, mask, discontinuity_map, pad_discontinuity_patches,
                        pad_discontinuity_hole_patches, window_size, pad_mask_patches, sigma_r, spatial_term):
    p_h, p_w = pad_depth_patches.shape[:2]
    for pi in range(p_h):
        for pj in range(p_w):
            if len(mask) != 0 and mask[pi, pj] == 0:
                continue
            if discontinuity_map is not None:
                if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                    continue
                discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]

            depth_patch = pad_depth_patches[pi, pj]
            depth_order = depth_patch.ravel().argsort()
            patch_midpt = depth_patch[window_size // 2, window_size // 2]
            
            if discontinuity_map is not None:
                coef = discontinuity_holes
                if len(mask) != 0:
                    coef = coef * pad_mask_patches[pi, pj]
            else:
                range_term = np.exp(-(depth_patch - patch_midpt) ** 2 / (2. * sigma_r ** 2))
                coef = (spatial_term * range_term).astype(np.float32)

            if np.max(coef) == 0:
                output[pi, pj] = patch_midpt
                continue

            if discontinuity_map is not None and (coef.max() == 0):
                output[pi, pj] = patch_midpt
            else:
                coef = coef / (coef.sum())
                coef_order = coef.ravel()[depth_order]
                cum_coef = np.cumsum(coef_order)

                value = 0.5
                if value < cum_coef[0]:
                    ind = 0

                if value >= cum_coef[len(cum_coef) - 1]:
                    ind = cum_coef.shape[0]

                for x in range(1, len(cum_coef)):
                    if cum_coef[x - 1] <= value < cum_coef[x]:
                        ind = x

                output[pi, pj] = depth_patch.ravel()[depth_order][ind]


@jit('void(float32[:,:],float32[:,:,:,:],float32[:,:],float32[:,:,:,:],int64,float64,float32[:,:])')
def _bilateral_filter_2(output, pad_depth_patches, discontinuity_map, pad_discontinuity_patches,
                        window_size, sigma_r, spatial_term):
    p_h, p_w = pad_depth_patches.shape[:2]
    for pi in range(p_h):
        for pj in range(p_w):
            if discontinuity_map is not None:
                if pad_discontinuity_patches[pi, pj][window_size // 2, window_size // 2] == 1:
                    continue
                discontinuity_patch = pad_discontinuity_patches[pi, pj]
                discontinuity_holes = (1. - discontinuity_patch)
            depth_patch = pad_depth_patches[pi, pj]
            depth_order = depth_patch.ravel().argsort()
            patch_midpt = depth_patch[window_size // 2, window_size // 2]
            range_term = np.exp(-(depth_patch - patch_midpt) ** 2 / (2. * sigma_r ** 2))
                
            if discontinuity_map is not None:
                coef = spatial_term * range_term * discontinuity_holes
            else:
                coef = spatial_term * range_term
                
            if coef.sum() == 0:
                output[pi, pj] = patch_midpt
                continue

            if discontinuity_map is not None and (coef.sum() == 0):
                output[pi, pj] = patch_midpt
            else:
                coef = coef / (coef.sum())
                coef_order = coef.ravel()[depth_order]
                cum_coef = np.cumsum(coef_order)

                value = 0.5
                if value < cum_coef[0]:
                    ind = 0

                if value >= cum_coef[len(cum_coef) - 1]:
                    ind = cum_coef.shape[0]

                for x in range(1, len(cum_coef)):
                    if cum_coef[x - 1] <= value < cum_coef[x]:
                        ind = x

                output[pi, pj] = depth_patch.ravel()[depth_order][ind]


def shape_fn(a, i, w, s):
    return (a.shape[i] - w) // s + 1


def acc_shape(a, i):
    if i + 1 >= len(a.shape):
        return 1
    else:
        return reduce(lambda x, y: x * y, a.shape[i + 1:])


def rolling_window(a, window, strides):
    assert len(a.shape) == len(window) == len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"

    shape = [shape_fn(a, i, w, s) for i, (w, s) in enumerate(zip(window, strides))] + list(window)
    _strides = [acc_shape(a, i) * s * a.itemsize for i, s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)


def sparse_bilateral_filtering(depth, image, mask=None, num_iter=1):
    save_images = []
    save_depths = []
    save_discontinuities = []
    vis_depth = depth.copy().astype(np.float32)

    for i in range(num_iter):
        window_size = get_window_size(idx=i)

        vis_image = image.copy()
        save_images.append(vis_image)
        save_depths.append(vis_depth)

        u_over, b_over, l_over, r_over = neighbor_depth_discontinuity(vis_depth, mask=mask)

        vis_image[u_over > 0] = np.array([0, 0, 0])
        vis_image[b_over > 0] = np.array([0, 0, 0])
        vis_image[l_over > 0] = np.array([0, 0, 0])
        vis_image[r_over > 0] = np.array([0, 0, 0])

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1

        save_discontinuities.append(discontinuity_map)

        if mask is not None:
            discontinuity_map[mask == 0] = 0

        vis_depth = bilateral_filter(vis_depth, discontinuity_map=discontinuity_map, mask=mask, window_size=window_size)

    return save_images, save_depths


#######################################################################################################################
# depth discontinuity
#######################################################################################################################


def init_neighbor_diff(disparity):
    u_diff = (disparity[1:, :] - disparity[:-1, :])[:-1, 1:-1]
    b_diff = (disparity[:-1, :] - disparity[1:, :])[1:, 1:-1]
    l_diff = (disparity[:, 1:] - disparity[:, :-1])[1:-1, :-1]
    r_diff = (disparity[:, :-1] - disparity[:, 1:])[1:-1, 1:]

    return u_diff, b_diff, l_diff, r_diff


def compute_neighbor_mask(mask):
    u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
    b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
    l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
    r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]

    return u_mask, b_mask, l_mask, r_mask


def diff_with_mask(u_diff, b_diff, l_diff, r_diff, u_mask, b_mask, l_mask, r_mask):
    u_diff = u_diff * u_mask
    b_diff = b_diff * b_mask
    l_diff = l_diff * l_mask
    r_diff = r_diff * r_mask

    return u_diff, b_diff, l_diff, r_diff


def compute_neighbor_diff(disparity, mask):
    u_diff, b_diff, l_diff, r_diff = init_neighbor_diff(disparity)

    if mask is not None:
        u_mask, b_mask, l_mask, r_mask = compute_neighbor_mask(mask)
        u_diff, b_diff, l_diff, r_diff = diff_with_mask(u_diff, b_diff, l_diff, r_diff, u_mask, b_mask, l_mask, r_mask)

    return u_diff, b_diff, l_diff, r_diff


def threshold_diff(u_diff, b_diff, l_diff, r_diff, threshold):
    u_over = np.array(np.abs(u_diff) > threshold).astype(np.float32)
    b_over = np.array(np.abs(b_diff) > threshold).astype(np.float32)
    l_over = np.array(np.abs(l_diff) > threshold).astype(np.float32)
    r_over = np.array(np.abs(r_diff) > threshold).astype(np.float32)

    return u_over, b_over, l_over, r_over


def diff_padding(u_diff, b_diff, l_diff, r_diff, u_over, b_over, l_over, r_over):
    u_over = np.pad(u_over, 1, mode='constant')
    b_over = np.pad(b_over, 1, mode='constant')
    l_over = np.pad(l_over, 1, mode='constant')
    r_over = np.pad(r_over, 1, mode='constant')
    u_diff = np.pad(u_diff, 1, mode='constant')
    b_diff = np.pad(b_diff, 1, mode='constant')
    l_diff = np.pad(l_diff, 1, mode='constant')
    r_diff = np.pad(r_diff, 1, mode='constant')

    return u_diff, b_diff, l_diff, r_diff, u_over, b_over, l_over, r_over


def neighbor_depth_discontinuity(depth, vis_diff=False, label=False, mask=None):
    if label is False:
        disparity = 1. / (depth + 1e-7)
        threshold = 0.02
    else:
        disparity = depth
        threshold = 0

    u_diff, b_diff, l_diff, r_diff = compute_neighbor_diff(disparity, mask)
    u_over, b_over, l_over, r_over = threshold_diff(u_diff, b_diff, l_diff, r_diff, threshold)

    u_diff, b_diff, l_diff, r_diff, u_over, b_over, l_over, r_over = diff_padding(u_diff, b_diff, l_diff, r_diff,
                                                                                  u_over, b_over, l_over, r_over)

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]
