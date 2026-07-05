# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op_common_random import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(
        self,
        offset=None,
        dtype=None,
        generator=None,
        high=None,
        low=None,
        seed=None,
        shape=None,
    ):
        dtype = self._dtype(dtype=dtype)
        offset_value = 0 if offset is None else int(np.asarray(offset).item())
        if generator not in (None, "unspecified"):
            res = self._deterministic_uniform(
                generator, seed, shape, dtype, offset_value
            )
            # low + r * (high - low), evaluated in the target data type
            low_t = np.asarray(low, dtype=dtype)
            high_t = np.asarray(high, dtype=dtype)
            res = res * (high_t - low_t) + low_t
        else:
            # The effect of offset on the values is implementation-defined
            # for the "unspecified" generator; it is ignored here.
            state = self._get_state(seed)
            res = state.rand(*shape).astype(dtype)
            res *= high - low
            res += low
            res = res.astype(dtype)
        if len(self.onnx_node.output) > 1:
            return (res, self._next_offset(offset_value))
        return (res,)
