# Tokenization

- Status: planned
- Date: TBD

> **Planned ADR — not yet written.** Placeholder for the *Tokenization* row of
> [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md).
> **Owning project:** `Tokenizers`. **Stance:** Reuse a tokenizer library (BPE / SentencePiece).
> Open question this ADR settles: which library (e.g. `Microsoft.ML.Tokenizers`) and how the
> pre-tokenizer pipeline is sourced.
> On completion, rename to `YYMMDD-tokenization.md` and set Status to `proposed`.

## Where this problem is isolated

Contained entirely in **`InferenceEngine.Tokenizers`**:

- **Lives here:** the vocabulary, BPE merge rules, the pre-tokenizer regex, and special-token
  handling.
- **Exposed as:** `ITokenizer` (defined in `Core`) — `Encode(text) -> ids` and
  `Decode(ids) -> text`. `Engine` calls only those two methods.
- **Contained change:** the tokenizer library choice and how the pre-tokenizer pipeline is
  sourced stay inside `Tokenizers`.

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
