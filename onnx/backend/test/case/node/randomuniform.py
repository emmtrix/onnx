# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def mersenne_twister_uniform(seed, shape, low=0.0, high=1.0):
    """Reference values for RandomUniform with generator="mersenne_twister".

    The operator specifies MT19937 seeded with ``init_genrand`` and doubles
    drawn with the 53-bit ``genrand_res53`` method. For scalar seeds below
    2**32 numpy's legacy ``RandomState`` implements exactly this algorithm,
    so it can serve as an independent reference here.
    """
    state = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    return low + state.random_sample(shape) * (high - low)


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

        y = mersenne_twister_uniform(42, (3, 4)).astype(np.float32)
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

        y = mersenne_twister_uniform(0, (2, 3), low=5.0, high=10.0).astype(np.float32)
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

        y = mersenne_twister_uniform(123, (2, 4))
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

        y = mersenne_twister_uniform(7, (10,)).astype(np.float16)
        expect(
            node,
            inputs=[],
            outputs=[y],
            name="test_randomuniform_mersenne_twister_float16",
        )
