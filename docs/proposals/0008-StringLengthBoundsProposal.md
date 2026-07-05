<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: `string_length_bounds`
- Start Date: 2026-07-05
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors: (to be filled in)

## Summary
[summary]: #summary

Add an optional `max_string_length` field to `TypeProto.Tensor` that declares
an upper bound, in bytes of the UTF-8 encoding, on the length of each string
element of a tensor. The bound is a static type property: it can be declared
on model inputs and value infos, it is validated by the checker, and shape
inference propagates it conservatively through the graph. Consumers that do
not need the bound can ignore it; consumers that require statically-sized
storage (for example, code generators for embedded targets) can rely on it.

## Motivation
[motivation]: #motivation

ONNX string tensors are unbounded: nothing in the model constrains how long
an individual string element may be. For most runtimes this is not a problem,
because strings are allocated dynamically. However, an important class of
consumers cannot allocate dynamically at all:

- **Code generators for embedded and safety-critical targets** translate an
  ONNX model into freestanding C/C++ code with fully static memory. Every
  buffer must have a compile-time size. For numeric tensors this is possible
  once symbolic dimensions are fixed; for string tensors it is impossible
  today, so such tools must reject any model containing a string tensor.
- **Ahead-of-time compilers and planners** that precompute worst-case memory
  usage face the same problem: the memory footprint of a string tensor is
  unknowable from the model.

String tensors appear in real, deployable models, most prominently in
preprocessing and postprocessing pipelines converted by sklearn-onnx and
similar converters: categorical features flow through operators such as
`LabelEncoder`, `CategoryMapper`, `OneHotEncoder`, and `TfIdfVectorizer`, and
newer opsets added general string manipulation (`StringConcat`, `StringSplit`,
`RegexFullMatch`). Today, deploying such a pipeline on a statically-allocated
target requires manually rewriting the model to remove strings.

In practice a sound upper bound on string lengths almost always exists and is
even derivable:

- The string values consumed or produced by the classical-ML operators come
  from vocabularies stored **inside the model** (attributes or initializers),
  whose maximum length is trivially computable.
- For genuine string *inputs*, the deployment context defines a bound (a
  fixed-width field in a message format, a `VARCHAR(n)` database column, a
  bounded sensor identifier). Declaring it is analogous to fixing a symbolic
  batch dimension before deployment — a step that embedded deployment flows
  already require for numeric tensors.

What is missing is a standard place to record this bound so that tools can
validate it, propagate it, and rely on it. This proposal adds that place to
the type system.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

A tensor type with string elements MAY declare a **string length bound**:

```proto
message Tensor {
  optional int32 elem_type = 1;
  optional TensorShapeProto shape = 2;
  // An optional upper bound on the length of each string element of the
  // tensor, measured in bytes of the UTF-8 encoding.
  optional int64 max_string_length = 3;
}
```

- The bound is measured in **bytes of the UTF-8 encoding**, not in code
  points, because its purpose is sizing storage.
- The field MAY be set only when `elem_type` is `STRING`, and MUST be a
  positive value; the checker enforces both.
- Every string element of a runtime value of the type MUST NOT be longer
  than the bound. A model whose runtime values violate a declared bound is
  invalid, exactly like a model whose runtime shapes contradict a declared
  static shape.
- When the field is absent, string lengths are unbounded — the meaning of
  every existing model is unchanged.

In the textual format the bound is written in parentheses after the `string`
type name:

```
agraph (string(64)[N] names) => (string(64)[N] out) {
   out = Identity(names)
}
```

The Python helpers accept the bound as an optional argument:

```python
vi = helper.make_tensor_value_info(
    "names", TensorProto.STRING, ["N"], max_string_length=64
)
```

**Shape inference propagates bounds.** A user only needs to annotate the
model boundary (typically graph inputs); inference derives bounds for
intermediate values conservatively — an inferred bound is always a valid
upper bound, and where soundness cannot be guaranteed the value simply stays
unbounded. For example, with a `string(8)` and a `string(100)` input,
`Concat` infers `string(100)` (elements are moved, not changed), while
`StringConcat` infers `string(108)` (elements are concatenated).

**Consumers may ignore the bound.** A runtime that allocates strings
dynamically can disregard the field entirely. A runtime MAY use it for
preallocation, and MAY treat a violating runtime value as an error. Tools
that require the bound (static code generators) reject unbounded string
tensors with a clear message, and accept bounded ones.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Proto change

One new optional field on `TypeProto.Tensor` (field number 3). As an
optional scalar field, it is wire-compatible in both directions: older
consumers preserve it as an unknown field, and models that do not use it are
byte-identical to today's encoding.

Use of the field is gated on the IR version that introduces it: the checker
MUST reject a model that sets `max_string_length` but declares an older
`ir_version`.

### Validation

`check_value_info` enforces:

- `max_string_length` set ⇒ `elem_type == STRING`, else fail.
- `max_string_length` set ⇒ value strictly positive, else fail.

### Shape inference

The propagation rules are chosen so that **an inferred bound is always
sound**; where soundness cannot be established, the output is left
unbounded (absence of a bound is always correct).

1. **Value-preserving operations** (the default rule, applied in the shared
   element-type propagation used by data-movement and selection operators
   such as `Identity`, `Reshape`, `Transpose`, `Concat`, `Split`, `Gather`,
   `Slice`, `Where`, `Pad`, `ScatterElements`, `StringSplit`): the output
   bound is the **maximum bound over all inputs**, provided every input that
   can carry string values — including strings nested in sequence, optional,
   and map types — declares a bound. If any string-carrying input is
   unbounded (or of unknown type), the output is unbounded. This is sound
   for any operator whose output strings are drawn from the string values of
   its inputs, which is why it can safely serve as the default.
2. **`StringConcat`**: the output bound is the **sum** of the two input
   bounds (with overflow protection), or absent if either input is
   unbounded.
3. **`CastLike`**: a bound propagated from the target-type input does not
   apply to values converted from the data input. The output keeps a bound
   only when the data input is itself a bounded string tensor (string-to-
   string cast preserves values), and is unbounded otherwise.
4. **String-creating operations without a derivable bound** (`Cast` to
   string from numeric types, `StringNormalizer` — Unicode case mapping can
   change the UTF-8 byte length): the output is unbounded.
5. **Type union** (merging the branches of `If`, loop-carried values): the
   union of two bounded string types is bounded by the **larger** bound; the
   union with an unbounded string type is unbounded.
6. **Merging inferred with declared types**: both bounds are valid upper
   bounds for the same value, so the **tighter** one is kept. A user
   annotation that is tighter than the inferred bound survives; a looser one
   is tightened.

Corner cases, by example:

- `Concat(string(8), string(100)) → string(100)`: elements are rearranged,
  not modified; the max rule is sound where a naive "propagate from first
  input" would be unsound (it would claim 8).
- `StringConcat(string(8), string(100)) → string(108)`: the max rule would
  be unsound here (a result can be 108 bytes long); the operator overrides
  the default with the sum rule.
- `CastLike(float_data, string(64)) → string` (unbounded): the result
  strings are decimal renderings of floats and have no relation to the
  target tensor's bound.
- `If` with branches yielding `string(8)` and `string(16)` → `string(16)`;
  with one unbounded branch → unbounded.
- A declared `string(32)` value info whose producer infers `string(64)`
  keeps `string(32)`; a declared `string(100)` is tightened to `string(64)`.

### Textual format

The grammar gains an optional length bound on the string element type:

```bnf
max-string-length ::= '(' int-constant ')'
tensor-elem-type  ::= prim-type | 'string' max-string-length
```

The printer emits the bound, so text round-trips are lossless.

### What this proposal does not change

- Operator signatures, operator semantics, and all existing type constraints
  are untouched; `tensor(string)` remains a single elementary type.
- Serialized size and wire compatibility of existing models are unchanged.
- Runtimes are not required to enforce the bound.

A complete implementation of everything described here (proto, checker,
helpers, parser/printer, shape inference, documentation, and tests)
accompanies this proposal.

## Drawbacks
[drawbacks]: #drawbacks

- **Exporters will not produce the field.** Source frameworks have no notion
  of bounded strings, so the bound will come from user annotation or from
  derivation tools, not from `torch.onnx` or tf2onnx. The feature's value
  is therefore concentrated on the (deliberate) deployment step rather than
  the export step.
- **Soundness requires per-operator review.** Every operator that creates
  string values (rather than moving them) must either derive a sound bound
  or suppress propagation. The conservative default limits the blast radius
  — an operator overlooked by the rules yields *no* bound, never a wrong
  one — except for operators whose outputs can be longer than all of their
  inputs' strings; those must override the default, and new string
  operators added in future opsets must be reviewed for this property.
- **A stale bound is a real constraint.** If a producer declares a bound
  that its data later violates, behavior is undefined at the consumer. This
  is the same class of risk as declaring a wrong static shape, but it is a
  new way to write an invalid model.
- **Spec surface grows** for a feature that a majority of runtimes will
  ignore.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- **`metadata_props` naming convention** (e.g.
  `"max_string_length.<value_name>": "64"`): requires no spec change, but
  the information is invisible to the type system — no checker validation,
  no propagation through shape inference, no textual syntax, and every
  consumer must reimplement the (subtle) propagation rules. Bounds also go
  stale silently when a graph is transformed and values are renamed.
- **Type denotation**: denotations are free-form semantic labels, not
  machine-checkable constraints with defined propagation semantics.
- **A family of fixed-width string element types** (`string16`, `string64`,
  …, in the spirit of NumPy's `<U16`): would multiply every string type
  constraint in every operator schema and break the closed enum of
  elementary types; a parameterized attribute of the tensor type is far
  less invasive.
- **Representing bounded strings as `uint8` tensors with an extra
  dimension**: expressible today, but changes the semantics of every
  operator that touches the value (a `Concat` on the last axis is not a
  string concat), and loses the distinction between "string of length ≤ n"
  and "byte matrix".
- **Impact of not doing this**: statically-allocating consumers must
  categorically reject string tensors, cutting off classical-ML
  preprocessing pipelines from embedded deployment, or invent proprietary
  side-channel formats for exactly this information.

## Prior art
[prior-art]: #prior-art

Length-bounded string types are long-established wherever storage must be
sized ahead of time:

- SQL `VARCHAR(n)` / `CHAR(n)`.
- NumPy fixed-width string dtypes (`<U16`, `S32`).
- HDF5 fixed-length string datatypes.
- Apache Arrow's fixed-size binary type.
- C/C++ `char[N]` fields in message and record layouts (the typical
  interface of the embedded systems this proposal targets).

Within ONNX itself, the closest analogue is the static shape: a runtime
property (the dynamic shape / the actual string length) is bounded by an
optional static declaration in the type, which shape inference propagates
and which consumers may exploit or ignore.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **IR version**: which IR version introduces the field, and the exact
  checker gating for models with older `ir_version` values.
- **Violation semantics**: this proposal defines a violating runtime value
  as making the model invalid (mirroring wrong static shapes). Whether
  runtimes SHOULD detect and report violations, or MAY exhibit undefined
  behavior, should be settled during the RFC discussion.
- **Scope of the field**: whether `TypeProto.SparseTensor` and map key
  types should carry the same bound. This proposal restricts the field to
  dense tensors to keep the surface minimal; the propagation rules already
  treat string content in other type kinds as unbounded, so extending the
  scope later is backward compatible.
- **Attribute-derived bounds**: operators whose output strings come from
  attributes (`LabelEncoder`, `CategoryMapper`, `Constant`,
  `ConstantOfShape`) could infer exact bounds from those attributes. This
  is sound and useful, but enlarges the inference surface; it can be added
  incrementally per operator.

## Future possibilities
[future-possibilities]: #future-possibilities

- **Automatic bound derivation**: a small tool (or a shape-inference
  extension, see above) that computes bounds from in-model string sources —
  classical-ML vocabulary attributes and string initializers. For typical
  sklearn-onnx pipelines this would make the entire model bounded with no
  user annotation at all.
- **Reference-implementation enforcement**: the reference evaluator could
  optionally verify declared bounds against actual values, catching stale
  annotations at test time.
- **Checker cross-validation of initializers**: string initializers whose
  declared value info carries a bound could be checked against it in
  `full_check` mode.
- **Bounds for sequences, maps, and sparse tensors**, should use cases
  emerge.
