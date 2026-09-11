# InferenceEngine.Core.Tests

`InferenceEngine.Core` is intentionally thin: it holds data contracts (`ModelConfig`,
`TokenizerData`) and interfaces (`IModel`, `ITokenizer`, `IKvCache`) with no logic of its own.
There is very little behavior here worth testing — most of what this layer promises is enforced by
the C# compiler (record shapes, interface members), not by runtime logic.

## Business scenarios covered

- **`ModelConfig` has real value semantics.** Two `ModelConfig` instances describing the same
  model compare equal, and a changed field makes them unequal. `ModelConfig`'s fields are all
  scalars, so the compiler-synthesized record equality already does the right thing.
- **`TokenizerData` value semantics — a documented gap, not a passing guarantee.** `TokenizerData`
  holds `string[]` fields (`Tokens`, `Merges`). C#'s synthesized record equality compares array
  fields by reference, not by content, so two `TokenizerData` instances built from identical vocab
  data are equal only if they share the exact same array instances — not if they were parsed
  independently from the same bytes. This is flagged as a production-code finding in
  `ValueSemanticsTests`, not silently patched, since fixing it (custom `IEquatable<T>` or a
  value-equatable collection type) is outside a test project's scope.

No other testable surface exists in this project today.
