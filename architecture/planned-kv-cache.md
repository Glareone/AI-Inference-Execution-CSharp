# KV-cache design

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *KV-cache* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Engine`. **Stance:** Build — start simple (contiguous), understand before
> optimizing (paged/quantized cache is a later step, if at all).
> On completion, rename to `YYMMDD-kv-cache.md` and set Status to `proposed`.

## Where this problem is isolated

Contained in **`InferenceEngine.Engine`**:

- **Lives here:** the key/value store, its growth/eviction policy, and the per-step update logic.
- **Exposed as:** an `IKvCache` abstraction (defined in `Core`) that `Models.Forward` receives and
  updates — the model doesn't own the cache memory or its policy.
- **Contained change:** moving from a simple contiguous cache to a paged/quantized one changes
  only `Engine`, not the forward pass.

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

## Decision Log

| Date       | Change              | By             |
|------------|---------------------|----------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
