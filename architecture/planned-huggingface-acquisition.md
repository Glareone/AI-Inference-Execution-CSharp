# HuggingFace model acquisition

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Model acquisition* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Cli` (hands `Models` a local path). **Stance:** Reuse — download once from HuggingFace into a
> local cache, reuse across runs. Open questions this ADR settles: which download library, the
> cache directory layout, and resume/verification behavior.
> On completion, rename to `YYMMDD-huggingface-acquisition.md` and set Status to `proposed`.

## Where this problem is isolated

Kept at the **`Cli`** edge, feeding **`Models`** a local path:

- **Lives here:** the download + local-cache logic sits at the `Cli` edge (a small acquisition
  helper); it resolves a repo/file reference to a **local file path**.
- **Exposed as:** that local path, handed to `Models.Load(path)`. `Models` only ever sees a path —
  it knows nothing about HTTP, HuggingFace, or the cache.
- **Contained change:** network and cache concerns stay out of the inference core; an offline run
  passes an already-cached path with no code change.

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
- [huggingface-ecosystem.md](../investigation/huggingface-ecosystem.md)

## Decision Log

| Date       | Change              | By             |
|------------|---------------------|----------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
