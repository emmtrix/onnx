# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op_common_random import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(
        self, dtype=None, generator=None, high=None, low=None, seed=None, shape=None
    ):
        dtype = self._dtype(dtype=dtype)
        if generator not in (None, "default"):
            res = self._deterministic_uniform(generator, seed, shape)
            res = res * (high - low) + low
            return (res.astype(dtype),)
        state = self._get_state(seed)
        res = state.rand(*shape).astype(dtype)
        res *= high - low
        res += low
        return (res.astype(dtype),)
