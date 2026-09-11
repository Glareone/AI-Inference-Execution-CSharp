# Sampling and logits pipeline

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Sampling / logits* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Engine`. **Stance:** Build — a composable temperature / top-k / top-p
> (and repetition-penalty) chain.
> On completion, rename to `YYMMDD-sampling-pipeline.md` and set Status to `proposed`.

## Where this problem is isolated

Contained in **`InferenceEngine.Engine`**:

- **Lives here:** a composable chain of steps over the logits `Span<float>` returned by
  `Forward` — temperature, top-k, top-p, repetition penalty — each step independent.
- **Exposed as:** sampling options on the `Engine` API. `Cli` only sets temperature/k/p and never
  sees logits; `Models` and `Tokenizers` don't know sampling exists.
- **Contained change:** adding, removing, or reordering steps is local to `Engine`.

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
