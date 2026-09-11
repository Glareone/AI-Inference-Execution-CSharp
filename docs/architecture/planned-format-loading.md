# Model format loading

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Model loading* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Models`. **Stance:** Reuse a parsing library; memory-map the weight blob
> (no copy). Open question this ADR settles: GGUF vs. SafeTensors, and which parsing library.
> On completion, rename to `YYMMDD-format-loading.md` and set Status to `proposed`.

## Where this problem is isolated

Contained entirely in **`InferenceEngine.Models`**:

- **Lives here:** the byte-level format parser (header, metadata, tensor descriptors) and the
  memory-mapped weight buffer. Nothing outside `Models` ever touches file offsets or raw bytes.
- **Exposed as:** `IModel` + `ModelConfig` (defined in `Core`). `Engine` receives a loaded
  `IModel` and cannot tell GGUF from SafeTensors.
- **Contained change:** swapping the format or the parsing library stays inside `Models` — no
  ripple into `Engine` or `Cli`.

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
- [gguf-format-research.md](../investigation/gguf-format-research.md)

## Decision Log

| Date       | Change              | By             |
|------------|---------------------|----------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
