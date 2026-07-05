// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for the random-generator operators in default domain from version
// 28 to 27 (currently RandomUniform; intended for the other random operators
// when they adopt the deterministic generator mechanism).

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class RandomGenerator_28_27 final : public Adapter {
 public:
  RandomGenerator_28_27(std::string op_name, size_t num_legacy_inputs)
      : Adapter(std::move(op_name), OpSetID(28), OpSetID(27)), num_legacy_inputs_(num_legacy_inputs) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    // The optional offset input (the first input after the operator's legacy
    // inputs) does not exist before version 28 and cannot be expressed in
    // older opsets. An omitted optional input may still be present as an
    // empty string, which the proto importer materializes as a kUndefined
    // placeholder; drop the placeholder so the node satisfies the old
    // schema's input arity.
    if (node->inputs().size() > num_legacy_inputs_) {
      ONNX_ASSERTM(
          node->inputs().size() == num_legacy_inputs_ + 1 &&
              node->inputs()[num_legacy_inputs_]->node()->kind() == kUndefined,
          "Operator '",
          name(),
          "' with an 'offset' input is not supported in Opset Version ",
          static_cast<int64_t>(target_version().version()),
          ".");
      node->removeInput(num_legacy_inputs_);
    }
    // seed_int64 does not exist before version 28.
    ONNX_ASSERTM(
        !node->hasAttribute(Symbol("seed_int64")),
        "Attribute 'seed_int64' of operator '",
        name(),
        "' is not supported in Opset Version ",
        static_cast<int64_t>(target_version().version()),
        ".");
    const Symbol generator("generator");
    if (node->hasAttribute(generator)) {
      // "unspecified" matches the implementation-defined behavior of the
      // pre-28 operators, so the attribute can simply be dropped. Any other
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

 private:
  size_t num_legacy_inputs_;
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
