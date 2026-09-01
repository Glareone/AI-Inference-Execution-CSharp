# Attention and transformer forward pass

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Transformer forward pass / attention*
> row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Models` (+ `Core` types). **Stance:** Build on top of a math library, no
> custom kernels. Covers embeddings, RMSNorm, RoPE, attention (MHA/GQA/MQA), SwiGLU FFN, and the
> projection to logits.
> On completion, rename to `YYMMDD-attention-and-transformer.md` and set Status to `proposed`.

## Where this problem is isolated

Owned by **`InferenceEngine.Models`**, using tensor types from **`Core`**:

- **Lives here:** the layer wiring (embeddings, RMSNorm, RoPE, GQA attention, SwiGLU FFN, LM
  head) and every call into the math library.
- **Exposed as:** `IModel.Forward(tokens, positions, kvCache) -> logits`. `Engine` never touches
  tensor internals beyond `Core`'s tensor types.
- **Boundary:** the KV-cache is passed *in* (owned by `Engine`) through a `Core` interface, so the
  forward pass and the cache stay independently swappable.

## Context and Problem Statement

_TBD._

## Decision Drivers

- _TBD_

## Considered Options

- _TBD_

## Decision Outcome

_TBD._

### Consequences

Legend: 🟢 upside · 🟡 accepted trade-off · 🔴 downside.

- _TBD_

## More Information

- [260811-solution-and-project-layout.md](260811-solution-and-project-layout.md)
- [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md)
- [dotllm-architecture-trace.md](../investigation/dotllm-architecture-trace.md)

## Decision Log

| Date       | Change              | By             |
|------------|---------------------|----------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
