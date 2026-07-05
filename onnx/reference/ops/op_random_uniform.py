# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
        seed_int64=None,
        shape=None,
    ):
        dtype = self._dtype(dtype=dtype)
        if generator not in (None, "unspecified"):
            return (
                self._deterministic_uniform(
                    generator, seed, seed_int64, shape, dtype, low, high, offset
                ),
            )
        # The effect of offset on the values is implementation-defined for
        # the "unspecified" generator; it is ignored here.
        state = self._get_state(seed)
        res = state.rand(*shape).astype(dtype)
        res *= high - low
        res += low
        return (res.astype(dtype),)
