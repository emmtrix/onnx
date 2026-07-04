# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op_common_random import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(
        self, dtype=None, generator=None, high=None, low=None, seed=None, shape=None
    ):
        dtype = self._dtype(dtype=dtype)
        if generator not in (None, "unspecified"):
            res = self._deterministic_uniform(generator, seed, shape, dtype)
            # low + r * (high - low), evaluated in the target data type
            low_t = np.asarray(low, dtype=dtype)
            high_t = np.asarray(high, dtype=dtype)
            return (res * (high_t - low_t) + low_t,)
        state = self._get_state(seed)
        res = state.rand(*shape).astype(dtype)
        res *= high - low
        res += low
        return (res.astype(dtype),)
