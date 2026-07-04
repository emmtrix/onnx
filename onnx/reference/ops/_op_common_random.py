# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


class _MT19937:
    """Standard 32-bit Mersenne Twister (MT19937).

    Implements the ``init_genrand`` seeding routine and the ``genrand_res53``
    double generation method from the reference implementation of Matsumoto
    and Nishimura (mt19937ar.c). The 32-bit output stream matches C++
    ``std::mt19937`` seeded with the same value. This is the algorithm
    selected by the ``generator="mersenne_twister"`` attribute of the random
    operators, which fully specifies their output for a given seed.
    """

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int):
        mt = [0] * self._N
        mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self._N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self._mt = mt
        self._index = self._N

    def _twist(self) -> None:
        mt = self._mt
        for i in range(self._N):
            y = (mt[i] & self._UPPER_MASK) | (mt[(i + 1) % self._N] & self._LOWER_MASK)
            mt[i] = (
                mt[(i + self._M) % self._N]
                ^ (y >> 1)
                ^ (self._MATRIX_A if y & 1 else 0)
            )
        self._index = 0

    def next_uint32(self) -> int:
        if self._index >= self._N:
            self._twist()
        y = self._mt[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def random_res53(self, num: int) -> np.ndarray:
        """Draw `num` doubles in [0, 1) with 53-bit resolution (genrand_res53)."""
        res = np.empty(num, dtype=np.float64)
        for k in range(num):
            a = self.next_uint32() >> 5
            b = self.next_uint32() >> 6
            res[k] = (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
        return res


class _CommonRandom(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        if hasattr(self, "shape") and len(self.shape) == 0:
            raise ValueError(  # pragma: no cover
                f"shape cannot be empty for operator {self.__class__.__name__}."
            )

    @staticmethod
    def numpy_type(dtype):
        return tensor_dtype_to_np_dtype(dtype)

    @staticmethod
    def _dtype(*data, dtype=None, dtype_first=False):
        numpy_type = _CommonRandom.numpy_type(dtype)
        if dtype_first and numpy_type is not None:
            if dtype != 0:
                return numpy_type
            if data:
                return data[0].dtype
            raise RuntimeError(
                f"dtype cannot be None for a random operator {_CommonRandom.__name__!r}, "
                f"numpy_type={numpy_type}, len(data)={len(data)}."
            )
        res = None
        if not data:
            res = numpy_type
        elif numpy_type is not None:
            res = numpy_type
        elif hasattr(data[0], "dtype"):
            res = data[0].dtype
        if res is None:
            raise RuntimeError(
                f"dtype cannot be None, numpy_type={numpy_type}, type(data[0])={type(data[0])}."
            )
        return res

    @staticmethod
    def _get_state(seed):
        if seed is None or np.isnan(seed):
            state = np.random.RandomState()
        else:
            state = np.random.RandomState(seed=int(seed))
        return state

    @staticmethod
    def _deterministic_uniform(generator, seed, shape):
        """Draw uniform doubles in [0, 1) with the fully specified generator.

        Unlike the "default" generator, the result is bit-identical across
        implementations for a given seed (see the operator specification).
        """
        if generator != "mersenne_twister":
            raise ValueError(
                f"Unsupported value {generator!r} for attribute 'generator' "
                f"(expected 'default' or 'mersenne_twister')."
            )
        if seed is None or np.isnan(seed):
            raise ValueError(
                "Attribute 'seed' must be specified when 'generator' is "
                "'mersenne_twister'."
            )
        state = _MT19937(int(seed) & 0xFFFFFFFF)
        num = int(np.prod(shape))
        return state.random_res53(num).reshape(shape)
