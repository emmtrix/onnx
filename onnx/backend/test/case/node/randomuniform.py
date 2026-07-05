# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def philox_uniform(seed, shape, dtype, low=0.0, high=1.0):
    """Independent implementation of RandomUniform with generator="philox4x32_10".

    Follows the operator specification: Philox-4x32-10 keyed with the 64-bit
    seed, counter block b = (lo32(b), hi32(b), 0, 0), per-element values in
    [0, 1) with a resolution matching the precision of `dtype` (two output
    words per element for double, one otherwise), and
    ``low + r * (high - low)`` evaluated in `dtype`. Kept separate from
    onnx.reference so the generated test data cross-checks the reference
    implementation.
    """
    m0, m1 = 0xD2511F53, 0xCD9E8D57
    w0, w1 = 0x9E3779B9, 0xBB67AE85
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    key0, key1 = seed & 0xFFFFFFFF, seed >> 32

    def block(b):
        c = [b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF, 0, 0]
        k0, k1 = key0, key1
        for r in range(10):
            if r > 0:
                k0 = (k0 + w0) & 0xFFFFFFFF
                k1 = (k1 + w1) & 0xFFFFFFFF
            p0 = m0 * c[0]
            p1 = m1 * c[2]
            c = [
                (p1 >> 32) ^ c[1] ^ k0,
                p1 & 0xFFFFFFFF,
                (p0 >> 32) ^ c[3] ^ k1,
                p0 & 0xFFFFFFFF,
            ]
        return c

    num = int(np.prod(shape))
    if np.dtype(dtype) == np.float64:
        r = []
        for i in range(num):
            w = block(i // 2)
            a, b = w[2 * (i % 2)], w[2 * (i % 2) + 1]
            r.append(((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0)
    else:
        p = ml_dtypes.finfo(dtype).nmant + 1
        r = [(block(i // 4)[i % 4] >> (32 - p)) / (1 << p) for i in range(num)]
    r = np.array(r, dtype=np.float64).reshape(shape).astype(dtype)
    low = np.asarray(low, dtype=dtype)
    high = np.asarray(high, dtype=dtype)
    return r * (high - low) + low


class RandomUniform(Base):
    @staticmethod
    def export_randomuniform_philox() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            shape=[3, 4],
            seed=42.0,
            generator="philox4x32_10",
        )

        y = philox_uniform(42, (3, 4), np.float32)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_philox",
        )

    @staticmethod
    def export_randomuniform_philox_low_high() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            low=5.0,
            high=10.0,
            shape=[2, 3],
            seed=0.0,
            generator="philox4x32_10",
        )

        y = philox_uniform(0, (2, 3), np.float32, low=5.0, high=10.0)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_philox_low_high",
        )

    @staticmethod
    def export_randomuniform_philox_double() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            dtype=onnx.TensorProto.DOUBLE,
            shape=[2, 4],
            seed=123.0,
            generator="philox4x32_10",
        )

        y = philox_uniform(123, (2, 4), np.float64)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_philox_double",
        )

    @staticmethod
    def export_randomuniform_philox_bfloat16() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            dtype=onnx.TensorProto.BFLOAT16,
            shape=[10],
            seed=3.0,
            generator="philox4x32_10",
        )

        y = philox_uniform(3, (10,), ml_dtypes.bfloat16)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_philox_bfloat16",
        )

    @staticmethod
    def export_randomuniform_philox_float16() -> None:
        node = onnx.helper.make_node(
            "RandomUniform",
            inputs=[],
            outputs=["y"],
            dtype=onnx.TensorProto.FLOAT16,
            shape=[10],
            seed=7.0,
            generator="philox4x32_10",
        )

        y = philox_uniform(7, (10,), np.float16)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_philox_float16",
        )
