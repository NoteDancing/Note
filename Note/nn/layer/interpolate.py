""" Interpolation helpers for Note layer

RegularGridInterpolator from https://github.com/sbarratt/torch_interpolations
Copyright NoteDance, Apache 2.0 license
"""
import tensorflow as tf
from itertools import product


class RegularGridInterpolator:
    """ Interpolate data defined on a rectilinear grid with even or uneven spacing.
    Produces similar results to scipy RegularGridInterpolator or interp2d
    in 'linear' mode.

    Taken from https://github.com/sbarratt/torch_interpolations
    """

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, tf.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, tf.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = tf.searchsorted(p, x, side="right")
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = tf.clip_by_value(idx_right - 1, 0, tf.shape(p)[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * \
                tf.reduce_prod(tf.stack(bs_s), axis=0)
        denominator = tf.reduce_prod(tf.stack(overalls), axis=0)
        return numerator / denominator