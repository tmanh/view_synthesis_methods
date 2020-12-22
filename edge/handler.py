import numpy as np

from noise.sparse_bilateral import sparse_bilateral_filtering


def smooth(image, depth, num_iter=5):
    _, bilateral = sparse_bilateral_filtering(depth, image, num_iter=num_iter)
    bilateral = bilateral[-1]

    return bilateral


def compute_edge(image, depth, num_iter=5):
    bilateral = smooth(image, depth, num_iter=num_iter)

    max_depth = bilateral.max()
    if max_depth >= 10000:
        max_depth = 5000
    norm_bilateral = (bilateral - bilateral.min()) / max(max_depth - bilateral.min(), 1e-7)

    horizon_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 0)) > 0.035).astype(np.uint8) * 255
    vertical_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 1)) > 0.035).astype(np.uint8) * 255
    check = np.maximum(horizon_check, vertical_check)

    return check, bilateral
