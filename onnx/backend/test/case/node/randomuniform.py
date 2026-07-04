# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def mersenne_twister_uniform(seed, shape, dtype, low=0.0, high=1.0):
    """Independent implementation of RandomUniform with generator="mersenne_twister".

    Follows the operator specification: MT19937 seeded with ``init_genrand``,
    per-element values in [0, 1) with a resolution matching the precision of
    `dtype` (two 32-bit outputs via ``genrand_res53`` for double, one 32-bit
    output otherwise), and ``low + r * (high - low)`` evaluated in `dtype`.
    Kept separate from onnx.reference so the generated test data cross-checks
    the reference implementation.
    """
    n, m = 624, 397
    mt = [0] * n
    mt[0] = int(seed) & 0xFFFFFFFF
    for i in range(1, n):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
    index = n

    def next_uint32():
        nonlocal index
        if index >= n:
            for i in range(n):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % n] & 0x7FFFFFFF)
                mt[i] = mt[(i + m) % n] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            index = 0
        y = mt[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    num = int(np.prod(shape))
    if np.dtype(dtype) == np.float64:
        r = [
            ((next_uint32() >> 5) * 67108864.0 + (next_uint32() >> 6))
            / 9007199254740992.0
            for _ in range(num)
        ]
    else:
        p = np.finfo(dtype).nmant + 1
        r = [(next_uint32() >> (32 - p)) / (1 << p) for _ in range(num)]
    r = np.array(r, dtype=np.float64).reshape(shape).astype(dtype)
    low = np.asarray(low, dtype=dtype)
    high = np.asarray(high, dtype=dtype)
    return r * (high - low) + low


class RandomUniform(Base):
    @staticmethod
    def export_randomuniform_mersenne_twister() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            shape=[3, 4],
            seed=42.0,
            generator="mersenne_twister",
        )

        y = mersenne_twister_uniform(42, (3, 4), np.float32)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_mersenne_twister",
        )

    @staticmethod
    def export_randomuniform_mersenne_twister_low_high() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            low=5.0,
            high=10.0,
            shape=[2, 3],
            seed=0.0,
            generator="mersenne_twister",
        )

        y = mersenne_twister_uniform(0, (2, 3), np.float32, low=5.0, high=10.0)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_mersenne_twister_low_high",
        )

    @staticmethod
    def export_randomuniform_mersenne_twister_double() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            dtype=onnx.TensorProto.DOUBLE,
            shape=[2, 4],
            seed=123.0,
            generator="mersenne_twister",
        )

        y = mersenne_twister_uniform(123, (2, 4), np.float64)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_mersenne_twister_double",
        )

    @staticmethod
    def export_randomuniform_mersenne_twister_float16() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            dtype=onnx.TensorProto.FLOAT16,
            shape=[10],
            seed=7.0,
            generator="mersenne_twister",
        )

        y = mersenne_twister_uniform(7, (10,), np.float16)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_mersenne_twister_float16",
        )
