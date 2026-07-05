# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import shape_inference


def _tensor_type_by_name(model: onnx.ModelProto, name: str) -> onnx.TypeProto.Tensor:
    for value_info in (
        list(model.graph.value_info)
        + list(model.graph.output)
        + list(model.graph.input)
    ):
        if value_info.name == name:
            return value_info.type.tensor_type
    raise AssertionError(f"No value info found for {name!r}")


def _infer(text: str) -> onnx.ModelProto:
    model = onnx.parser.parse_model(text)
    return shape_inference.infer_shapes(model)


class TestStringLengthInference(unittest.TestCase):
    def test_propagates_through_data_movement_ops(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(64)[4] X, int64[2] shape) => (string[N, M] Y)
            {
               T1 = Identity(X)
               T2 = Reshape(T1, shape)
               Y = Identity(T2)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "T1").max_string_length, 64)
        self.assertEqual(_tensor_type_by_name(inferred, "T2").max_string_length, 64)
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 64)

    def test_concat_uses_largest_input_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(8)[2] X, string(100)[3] Y) => (string[5] Z)
            {
               Z = Concat <axis = 0> (X, Y)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Z").max_string_length, 100)

    def test_no_bound_when_any_string_input_is_unbounded(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(8)[2] X, string[3] Y) => (string[5] Z)
            {
               Z = Concat <axis = 0> (X, Y)
            }
            """
        )
        self.assertFalse(
            _tensor_type_by_name(inferred, "Z").HasField("max_string_length")
        )

    def test_where_uses_largest_input_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (bool[3] cond, string(16)[3] X, string(32)[3] Y) => (string[3] Z)
            {
               Z = Where(cond, X, Y)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Z").max_string_length, 32)

    def test_string_concat_adds_bounds(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(8)[3] X, string(100)[3] Y) => (string[3] Z)
            {
               Z = StringConcat(X, Y)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Z").max_string_length, 108)

    def test_string_concat_unbounded_input(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(8)[3] X, string[3] Y) => (string[3] Z)
            {
               Z = StringConcat(X, Y)
            }
            """
        )
        self.assertFalse(
            _tensor_type_by_name(inferred, "Z").HasField("max_string_length")
        )

    def test_string_split_keeps_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(40)[3] X) => (string[3, N] Y, int64[3] Z)
            {
               Y, Z = StringSplit <delimiter = ","> (X)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 40)

    def test_cast_to_string_has_no_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (float[3] F) => (string[3] Z)
            {
               Z = Cast <to = 8> (F)
            }
            """
        )
        self.assertFalse(
            _tensor_type_by_name(inferred, "Z").HasField("max_string_length")
        )

    def test_cast_like_target_bound_does_not_apply(self) -> None:
        # The bound of the target-type input must not be applied to values
        # converted from a non-string input.
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (float[3] F, string(64)[3] S) => (string[3] Z)
            {
               Z = CastLike(F, S)
            }
            """
        )
        self.assertFalse(
            _tensor_type_by_name(inferred, "Z").HasField("max_string_length")
        )

    def test_cast_like_string_to_string_keeps_value_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(16)[3] S16, string(64)[3] S64) => (string[3] Z)
            {
               Z = CastLike(S16, S64)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Z").max_string_length, 16)

    def test_if_takes_largest_branch_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (bool cond, string(8)[3] A, string(16)[3] B) => (string[3] Y)
            {
               Y = If (cond) <
                  then_branch = tg () => (string[3] ty) { ty = Identity(A) },
                  else_branch = eg () => (string[3] ey) { ey = Identity(B) }>
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 16)

    def test_if_with_unbounded_branch(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (bool cond, string(8)[3] A, string[3] B) => (string[3] Y)
            {
               Y = If (cond) <
                  then_branch = tg () => (string[3] ty) { ty = Identity(A) },
                  else_branch = eg () => (string[3] ey) { ey = Identity(B) }>
            }
            """
        )
        self.assertFalse(
            _tensor_type_by_name(inferred, "Y").HasField("max_string_length")
        )

    def test_merge_keeps_tighter_declared_bound(self) -> None:
        # A declared bound on an intermediate value that is tighter than the
        # inferred one is kept.
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(64)[3] X) => (string[3] Y)
            <string(32)[3] T>
            {
               T = Identity(X)
               Y = Identity(T)
            }
            """
        )
        inferred = shape_inference.infer_shapes(model)
        self.assertEqual(_tensor_type_by_name(inferred, "T").max_string_length, 32)
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 32)

    def test_merge_tightens_declared_bound(self) -> None:
        # A declared bound that is looser than the inferred one is tightened.
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(64)[3] X) => (string[3] Y)
            <string(100)[3] T>
            {
               T = Identity(X)
               Y = Identity(T)
            }
            """
        )
        inferred = shape_inference.infer_shapes(model)
        self.assertEqual(_tensor_type_by_name(inferred, "T").max_string_length, 64)
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 64)

    def test_gather_keeps_bound(self) -> None:
        inferred = _infer(
            """
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (string(24)[10] X, int64[2] I) => (string[2] Y)
            {
               Y = Gather(X, I)
            }
            """
        )
        self.assertEqual(_tensor_type_by_name(inferred, "Y").max_string_length, 24)


if __name__ == "__main__":
    unittest.main()
