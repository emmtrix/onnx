// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for RandomUniform in default domain from version 28 to 27

#pragma once

#include <memory>
#include <string>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class RandomUniform_28_27 final : public Adapter {
 public:
  RandomUniform_28_27() : Adapter("RandomUniform", OpSetID(28), OpSetID(27)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    // The offset input does not exist in RandomUniform v22 and cannot be
    // expressed in older opsets.
    ONNX_ASSERTM(
        node->inputs().empty(),
        "Operator '",
        name(),
        "' with an 'offset' input is not supported in Opset Version ",
        static_cast<int64_t>(target_version().version()),
        ".");
    // seed_int64 does not exist in RandomUniform v22.
    ONNX_ASSERTM(
        !node->hasAttribute(Symbol("seed_int64")),
        "Attribute 'seed_int64' of operator '",
        name(),
        "' is not supported in Opset Version ",
        static_cast<int64_t>(target_version().version()),
        ".");
    const Symbol generator("generator");
    if (node->hasAttribute(generator)) {
      // "unspecified" matches the implementation-defined behavior of
      // RandomUniform v22, so the attribute can simply be dropped. Any other
      // generator selects fully specified deterministic output, which older
      // versions cannot express.
      ONNX_ASSERTM(
          node->s(generator) == "unspecified",
          "Attribute 'generator' of operator '",
          name(),
          "' must be 'unspecified' in Opset Version ",
          static_cast<int64_t>(target_version().version()),
          ".");
      node->removeAttribute(generator);
    }
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
