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
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"
#include "onnx/version_converter/helper.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class RandomGenerator_28_27 final : public Adapter {
 public:
  RandomGenerator_28_27(std::string op_name, size_t num_legacy_inputs)
      : Adapter(std::move(op_name), OpSetID(28), OpSetID(27)), num_legacy_inputs_(num_legacy_inputs) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // The optional offset input (the first input after the operator's legacy
    // inputs) does not exist before version 28. It can be removed without
    // changing semantics when it is omitted — possibly spelled as an empty
    // string, which the proto importer materializes as a kUndefined
    // placeholder — or when it is a constant 0, the documented pattern for
    // storing the stream position in the model. Any other offset selects a
    // stream that older versions cannot express.
    if (node->inputs().size() > num_legacy_inputs_) {
      ONNX_ASSERTM(
          node->inputs().size() == num_legacy_inputs_ + 1 && IsRemovableOffset(graph, node),
          "Operator '",
          name(),
          "' with an 'offset' input is not supported in Opset Version ",
          static_cast<int64_t>(target_version().version()),
          " (only an omitted offset or a constant offset of 0 can be removed).");
      RemoveOffsetInput(graph, node);
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

  bool IsRemovableOffset(const std::shared_ptr<Graph>& graph, Node* node) const {
    const Value* offset_val = node->inputs()[num_legacy_inputs_];
    const Node* offset_node = offset_val->node();
    if (offset_node->kind() == kUndefined) {
      return true;
    }
    if (offset_node->kind() == kConstant) {
      const std::vector<int64_t> values = ReadInt64Tensor(offset_node->t(kvalue));
      return values.size() == 1 && values[0] == 0;
    }
    if (graph->is_constant_initializer(offset_val)) {
      for (const auto& initializer : graph->initializers()) {
        if (initializer.name() == offset_val->uniqueName()) {
          const std::vector<int64_t> values = ReadInt64Tensor(initializer);
          return values.size() == 1 && values[0] == 0;
        }
      }
    }
    return false;
  }

  void RemoveOffsetInput(const std::shared_ptr<Graph>& graph, Node* node) const {
    Value* offset_val = node->inputs()[num_legacy_inputs_];
    Node* offset_node = offset_val->node();
    const std::string initializer_name = offset_val->uniqueName();
    const bool is_initializer = graph->is_constant_initializer(offset_val);
    node->removeInput(num_legacy_inputs_);
    if (offset_val->uses().empty()) {
      if (is_initializer) {
        graph->eraseInitializer(initializer_name);
      } else if (offset_node->kind() == kConstant) {
        offset_node->destroy();
      }
    }
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
