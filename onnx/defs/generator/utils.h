// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

void ConstantOpInference(InferenceContext& ctx);

// Shared documentation for the deterministic random-generator mechanism
// (generator / seed / seed_int64 attributes and the offset input), introduced
// with RandomUniform-28 and intended to be reused verbatim when the other
// random operators adopt the same mechanism.
inline constexpr const char* kRandomGeneratorSeedAttrDoc =
    "(Optional) Seed to the random generator, if not specified we will auto generate one. "
    "Used only when `generator` is \"unspecified\" (with implementation-defined effect); must not "
    "be specified together with a deterministic generator, which uses `seed_int64` instead.";

inline constexpr const char* kRandomGeneratorSeedInt64AttrDoc =
    "(Optional) 64-bit seed for the fully specified generators; its two's complement bits are "
    "interpreted as an unsigned 64-bit integer. Must be specified when `generator` is "
    "\"philox4x32_10\" (the float `seed` attribute is not used in that case). When `generator` is "
    "\"unspecified\", the effect of `seed_int64` is implementation-defined.";

inline constexpr const char* kRandomGeneratorAttrDoc =
    "(Optional) The pseudo-random number generator algorithm: \"unspecified\" leaves the choice of "
    "generator to the implementation and provides no determinism guarantee, even when a seed is "
    "specified; \"philox4x32_10\" selects the fully specified Philox-4x32-10 counter-based algorithm "
    "described in the operator documentation, making the output deterministic for a given "
    "`seed_int64`. More algorithms may be added in future opset versions.";

inline constexpr const char* kRandomGeneratorOffsetInputDoc =
    "(Optional) Scalar 64-bit stream offset, 0 if not provided. Each offset value selects an "
    "independent random stream (see the operator documentation for the exact semantics): feed a "
    "different offset in every run (any non-repeating scheme works, e.g. a step counter) to draw "
    "fresh, yet reproducible, values per run, or feed a constant (or omit the input) to draw the "
    "same values in every run. When `generator` is \"unspecified\", the effect of `offset` on the "
    "generated values is implementation-defined.";

// Validates the deterministic random-generator attributes: `generator` must
// be a known algorithm, deterministic generators require `seed_int64` and
// forbid the float `seed`, and the optional offset input (identified by
// `offset_input_index`, or -1 if the operator has none) must be a scalar.
void ValidateRandomGeneratorAttributes(InferenceContext& ctx, int offset_input_index);

template <typename T>
int64_t compute_output_dim_for_range(const TensorProto* start, const TensorProto* limit, const TensorProto* delta) {
  if (!start->dims().empty() || !limit->dims().empty() || !delta->dims().empty()) {
    fail_shape_inference("Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto start_data = ParseData<T>(start);
  const auto limit_data = ParseData<T>(limit);
  const auto delta_data = ParseData<T>(delta);

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  n = std::max<int64_t>(n, 0);

  return n;
}

} // namespace ONNX_NAMESPACE
