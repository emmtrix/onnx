# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


class _Philox4x32:
    """Philox-4x32-10 counter-based PRNG.

    Implements the Philox-4x32 generator with 10 rounds and the standard
    constants from Salmon et al., "Parallel random numbers: as easy as
    1, 2, 3" (SC'11), as also distributed in the Random123 library. This is
    the algorithm selected by the ``generator="philox4x32_10"`` attribute of
    the random operators, which fully specifies their output for a given
    seed. Being counter-based, every output word depends only on the key
    (derived from the seed) and the block index, so elements can be computed
    independently and in parallel.
    """

    _M0 = 0xD2511F53
    _M1 = 0xCD9E8D57
    _W0 = 0x9E3779B9
    _W1 = 0xBB67AE85

    def __init__(self, seed: int):
        seed &= 0xFFFFFFFFFFFFFFFF
        self._key0 = seed & 0xFFFFFFFF
        self._key1 = seed >> 32

    @classmethod
    def philox4x32_10(cls, c0, c1, c2, c3, key0: int, key1: int):
        """Encrypt 128-bit counters (four uint32 arrays) with 10 Philox rounds.

        Returns the four 32-bit output words per counter. The round keys start
        at ``(key0, key1)`` and are incremented by ``(W0, W1)`` before every
        round except the first.
        """
        mask = np.uint64(0xFFFFFFFF)
        c0 = np.asarray(c0, dtype=np.uint64)
        c1 = np.asarray(c1, dtype=np.uint64)
        c2 = np.asarray(c2, dtype=np.uint64)
        c3 = np.asarray(c3, dtype=np.uint64)
        k0, k1 = key0, key1
        for r in range(10):
            if r > 0:
                k0 = (k0 + cls._W0) & 0xFFFFFFFF
                k1 = (k1 + cls._W1) & 0xFFFFFFFF
            p0 = np.uint64(cls._M0) * c0
            p1 = np.uint64(cls._M1) * c2
            c0, c1, c2, c3 = (
                (p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0),
                p1 & mask,
                (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1),
                p0 & mask,
            )
        return (
            c0.astype(np.uint32),
            c1.astype(np.uint32),
            c2.astype(np.uint32),
            c3.astype(np.uint32),
        )

    def _blocks(self, num_blocks: int):
        """Output words of counter blocks 0 .. num_blocks-1.

        Block ``b`` uses the counter ``(lo32(b), hi32(b), 0, 0)``.
        """
        b = np.arange(num_blocks, dtype=np.uint64)
        zero = np.zeros(num_blocks, dtype=np.uint64)
        return self.philox4x32_10(
            b & np.uint64(0xFFFFFFFF),
            b >> np.uint64(32),
            zero,
            zero,
            self._key0,
            self._key1,
        )

    def random_res53(self, num: int) -> np.ndarray:
        """Draw `num` doubles in [0, 1) with 53-bit resolution.

        Element `i` combines words ``2*(i mod 2)`` and ``2*(i mod 2) + 1`` of
        block ``i // 2`` as ``(floor(a / 2^5) * 2^26 + floor(b / 2^6)) / 2^53``.
        """
        num_blocks = (num + 1) // 2
        w0, w1, w2, w3 = self._blocks(num_blocks)
        a = np.stack([w0, w2], axis=1).reshape(-1)[:num] >> np.uint32(5)
        b = np.stack([w1, w3], axis=1).reshape(-1)[:num] >> np.uint32(6)
        return (a.astype(np.float64) * 67108864.0 + b.astype(np.float64)) * (
            1.0 / 9007199254740992.0
        )

    def random_res(self, num: int, precision: int) -> np.ndarray:
        """Draw `num` values in [0, 1) with `precision` significand bits.

        Element `i` uses word ``i mod 4`` of block ``i // 4``:
        ``(w >> (32 - p)) / 2^p``. The results are exactly representable in
        any binary float type with at least `precision` significand bits.
        """
        num_blocks = (num + 3) // 4
        w0, w1, w2, w3 = self._blocks(num_blocks)
        words = np.stack([w0, w1, w2, w3], axis=1).reshape(-1)[:num]
        scale = 1.0 / (1 << precision)
        return (words >> np.uint32(32 - precision)).astype(np.float64) * scale


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
    def _deterministic_uniform(generator, seed, shape, dtype):
        """Draw uniform values in [0, 1) with the fully specified generator.

        Unlike the "unspecified" generator, the result is bit-identical across
        implementations for a given seed (see the operator specification).
        The resolution of the values matches the precision of `dtype`: double
        combines two 32-bit output words per element, all other float types
        use one word per element, keeping every value exactly representable
        in `dtype`.
        """
        if generator != "philox4x32_10":
            raise ValueError(
                f"Unsupported value {generator!r} for attribute 'generator' "
                f"(expected 'unspecified' or 'philox4x32_10')."
            )
        if seed is None or np.isnan(seed):
            raise ValueError(
                "Attribute 'seed' must be specified when 'generator' is "
                "'philox4x32_10'."
            )
        state = _Philox4x32(int(seed))
        num = int(np.prod(shape))
        if np.dtype(dtype) == np.float64:
            res = state.random_res53(num)
        else:
            precision = np.finfo(dtype).nmant + 1
            res = state.random_res(num, precision)
        return res.reshape(shape).astype(dtype)
