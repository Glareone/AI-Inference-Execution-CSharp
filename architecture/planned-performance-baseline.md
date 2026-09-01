# Performance baseline and expectations

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Performance* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** cross-cutting. **Stance:** Measure against dotLLM; managed-first, `unsafe`
> only with explicit sign-off (per AGENTS.md). Sets the tokens/second target and the rules for
> when a performance experiment justifies leaving the managed path.
> On completion, rename to `YYMMDD-performance-baseline.md` and set Status to `proposed`.

## Where this problem is isolated

**Cross-cutting — deliberately *not* isolated to one project:**

- **A measurement + policy concern:** the benchmark harness and data live in `experiments/`; the
  policy (managed-first, `unsafe` only with explicit sign-off) constrains every project.
- **Boundary that keeps it contained:** any hot-path optimization stays behind the same `Core`
  interfaces (`IModel.Forward`, `IKvCache`), so a faster implementation never changes the
  contracts other layers depend on.
- **Why here anyway:** it sets the shared target (tokens/second vs. dotLLM) that the other ADRs'
  implementation choices are measured against.

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
- [reference-measurements-dotllm.md](../experiments/reference-measurements-dotllm.md)

## Decision Log

| Date       | Change              | By             |
|------------|---------------------|----------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
