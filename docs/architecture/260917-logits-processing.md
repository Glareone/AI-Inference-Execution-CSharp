# Logits processing and sampling pipeline

- Status: proposed
- Date: 2026-09-17

## Where this problem is isolated

Contained in **`InferenceEngine.Engine`**, entirely within the existing `Sampling/` subfolder:

- **Lives here:** a new `ILogitsProcessor` interface and its first implementation
  (`BannedSequenceLogitsProcessor`), alongside the existing `ISamplerStep`/`SamplingPipeline` —
  same project, same namespace root (`InferenceEngine.Engine.Sampling`), distinct types. `Core`
  and `Tokenizers` are untouched; `Engine` already references `Tokenizers` for its live
  `ITokenizer` instance, so tokenizing ban strings adds no new project dependency.
- **Exposed as:** a `BannedWords` option on `GenerationOptions` (`Engine.Config`), consumed by a
  new CLI flag/env var. `Cli` still never sees logits directly — it only ever sets options and
  reads generated text, same boundary as today.
- **Contained change:** `SamplingPipeline.Sample`'s signature changes (adds
  `ReadOnlySpan<int> generatedTokenIds`) and its body gains an unconditional pre-branch step; no
  other project's public surface changes.

## Context and Problem Statement

The generation loop's `Forward` → sample → decode chain currently draws a two-way line
(tokenizer vs. sampler). Adding a ban mechanism (see below) makes it clear a third concept has
been hiding, unnamed, the whole time. This ADR draws the line four ways and gives the missing
piece a name:

- **Tokenizer** — text ↔ token-id translation. `InferenceEngine.Tokenizers`
  (`GgufBpeTokenizer`), already correctly separated into its own project. Unchanged by this ADR.
- **Logits** — **not a component, a data artifact.** The raw, unnormalized per-vocabulary-token
  scores produced once per `Forward` call by `InferenceEngine.Models`'s LM head. They're owned by
  nobody in particular — just data crossing the `Models → Engine` boundary via the `IModel`
  contract already defined in `Core`. It's worth being explicit about a natural misconception
  here: logits are not "deep inside the transformer" — they are the transformer's *output*, the
  very last thing it produces before anything downstream touches them. A logits processor
  therefore sits *after* the model and *before* sampling: the least "deep" position in the whole
  pipeline, not the most.
- **Logits Processor** — deterministic, rule-based transformations applied to raw logits
  unconditionally, before any randomness enters the picture: hard constraints such as banning a
  sequence of tokens outright (the concrete need this round), with repetition penalty, forced
  tokens, and grammar/schema-constrained decoding named as future seams (see Design). This is
  genuinely new territory — nothing resembling it exists anywhere in this codebase today
  (exhaustively grepped: the only per-token special case in the whole generation loop is the EOS
  check at `InferenceSession.cs:153`).
- **Sampler** — the existing `ISamplerStep`/`SamplingPipeline`
  (`src/InferenceEngine.Engine/Sampling/`): probabilistic distribution-shaping
  (temperature/top-k/top-p) that only matters when *not* doing greedy decoding. Unchanged in
  responsibility by this ADR.

The structural fact that makes this a real design question, not just a naming exercise:
`SamplingPipeline`'s constructor only populates `_steps` when `options.Temperature > 0`
(`SamplingPipeline.cs:18-30`). `Sample()` has two entirely separate code paths — a greedy argmax
scan (`Sample()`, lines ~35-47) that never touches `_steps` at all, and a stochastic path
(~49-83) that does. `Temperature <= 0` is the default in both `GenerationOptions` and the CLI, so
**the common case never consults `_steps`.** Anything that must apply unconditionally — a ban is
exactly this, it has to hold whether the caller is doing greedy or stochastic decoding — cannot be
implemented as "just another `ISamplerStep`"; it has to run before the greedy/stochastic branch
splits, which the current contract has no place for.

Multi-token banning is a real functional requirement, not a nice-to-have: a single banned word
like "EPAM" will almost certainly BPE-split into more than one token, so a single-token-only ban
mechanism would not actually ban it. This needs sequence-prefix matching, the same shape as
HuggingFace transformers' `NoBadWordsLogitsProcessor` (see More Information): a length-1 banned
sequence is always masked; a longer one is masked only when the tail of the tokens already
generated this turn matches the sequence's prefix.

This ADR replaces `docs/architecture/planned-sampling-pipeline.md` (Status: `planned`, every
section `_TBD._`) and supersedes its stated stance — "Build — a composable temperature/top-k/
top-p (and repetition-penalty) chain," owning project `Engine` — with the four-way split above
(see Decision Log). The placeholder undersold the shape of the problem: it treated
"repetition-penalty" as just another chain link alongside temperature/top-k/top-p, when in fact
it (and banning) belong to a different phase — deterministic and unconditional — than the
probabilistic steps they'd sit next to.

One deliberate non-change: `PrefillTopLogits` (`InferenceSession.cs`, the `--debug-logits` CLI
path) bypasses `SamplingPipeline` entirely today and does its own raw top-N logit extraction. This
ADR leaves it that way — it's a deliberate diagnostic view of *unprocessed* model output, and
running it through a logits processor or the sampler would defeat its purpose.

## Decision Drivers

- **No speculative abstractions (AGENTS.md).** Don't stand up a project, a `.csproj`, a new error
  hierarchy, or configuration surface for cases that don't exist yet. The only concrete new
  content on the table this round is sequence-based banning plus a small number of *named* future
  seams — not a general plugin architecture.
- **A new project needs real, non-speculative content; it isn't pre-approved by
  [260811](260811-solution-and-project-layout.md).** That ADR fixed five projects deliberately
  scoped to what had actual content at the time; adding a sixth is a decision this ADR has to
  earn on its own, not inherit by analogy to `Tokenizers`.
- **Alignment with industry precedent.** How other engines draw this exact line is directly
  relevant evidence, not decoration — see the survey below.
- **The concrete new content actually on the table.** Sequence-based banning (needs
  `ReadOnlySpan<int>` generation history, which `ISamplerStep.Apply(Span<float>)` doesn't carry)
  and a small set of named seams (repetition penalty, forced-token bias, grammar constraints).
  This is what makes the ask real rather than a purely speculative "let's have an
  `ILogitsProcessor` in case we need one someday."

## Considered Options

- **A — New top-level project `InferenceEngine.LogitsProcessing`.** Mirrors how `Tokenizers`
  already got its own project: a new `.csproj`, a new solution-file entry (GUID, Debug/Release
  config rows, nesting under the `src` solution folder in `InferenceEngine.sln`), and
  cross-reference updates across [260811](260811-solution-and-project-layout.md), `CLAUDE.md`,
  `CHANGELOG.md`, and `.claude/skills/csharp-conventions/SKILL.md` (whose frontmatter hard-codes
  "the five InferenceEngine.* projects").
- **B — Explicit internal split inside `Engine`.** A new `ILogitsProcessor` interface, distinct
  from `ISamplerStep`, in its own subfolder/namespace under `InferenceEngine.Engine.Sampling`.
  `SamplingPipeline` gains an unconditional pre-step that runs every registered processor before
  the greedy/stochastic branch. No new project, no new `.csproj`, no solution-file changes.
- **C — No new abstraction.** Stretch `ISamplerStep`'s existing signature to add a token-history
  parameter, and special-case a "must always run" step into the greedy path too, reusing
  `ISamplerStep` for both phases.

### Industry precedent (evidence for the decision below)

How other engines actually draw this line — surveyed before choosing, not after:

| Engine | Where "logits processing" lives | Separate project/package from the sampler? |
|---|---|---|
| **dotLLM** (our named primary reference) | `ISamplerStep`/`SamplerPipeline` inside its `Engine`-equivalent project. `SamplerPipeline.Sample()` runs step 1 "run logit processors (repetition penalty)", then step 3 "run sampler steps: temperature → top-k → top-p → min-p" (see `docs/investigation/dotllm-architecture-trace.md`, `SamplerPipeline.Sample()`) | **No** — two phases of one pipeline, one project. dotLLM's 17-project layout splits out `Tokenizers` but has no separate logits/sampling project. |
| **HuggingFace transformers** | `LogitsProcessorList` — a chain of `LogitsProcessor` objects (e.g. `NoBadWordsLogitsProcessor`, forbidding exact token-id *sequences* by overwriting their logits to `-inf`, checked against generation history), applied inside `generate()` before the sampling step | Conceptually distinct class in the same `generation` module — not a separate package |
| **vLLM** | `LogitsProcessor` interface (`apply(logits)`, `is_argmax_invariant()`, `update_state(batch_update)`) lives inside `vllm/v1/sample/` — the *same* module as the sampler. vLLM's own design docs admit bad-words/repetition-penalty are still "hard-coded into the sampler" and "not yet utilizing the programming model" — explicitly named architectural debt | **No** — same module; vLLM hasn't even fully separated it internally yet |
| **llama.cpp** | A single `llama_sampler_chain` (linked list of `llama_sampler` objects); `logit_bias` is just the *first link* in that chain, applied before penalties/top-k/top-p/temperature | **No** — one chain, one library; `logit_bias` isn't a separate module |

The pattern across every engine surveyed: **logits processing and sampling are a phase split
within one module/pipeline, never a project or package boundary.** This is weighed as evidence
below, not used to skip evaluating Option A on its own terms.

## Decision Outcome

Chosen option: **B — explicit internal split inside `Engine`**, because it is the only option
that both (a) matches how every engine surveyed actually draws this line, and (b) gives banning
the unconditional-execution guarantee it needs without misusing an existing contract to get it.

Option A is rejected, respectfully rather than dismissively — it was the option raised first, and
the `Tokenizers` precedent it appeals to is real. But its ceremony (a `.csproj`, a solution-file
entry, four documents' worth of cross-reference updates) isn't justified by what's actually being
built this round: one processor type, one concrete implementation, and a factory method. Standing
up a project for that would itself be exactly the kind of speculative scaffolding
[260811](260811-solution-and-project-layout.md)'s own Consequences warn against ("if we later *do*
want a custom CPU/CUDA kernel experiment or a server, we'll add those projects then — this ADR
doesn't pre-approve that; it'll need its own decision when it has real content"). The industry
survey reinforces this from a different angle: not one of the four engines surveyed — including
vLLM, which does have a first-class `LogitsProcessor` abstraction — treats logits processing as a
separate package from the sampler. vLLM is the most telling data point precisely because it's the
most sophisticated: it still keeps the two in the same module, and openly calls the gap between
"has an abstraction" and "actually decoupled" architectural debt. If a project with vLLM's scale
and resources hasn't found a reason to make this a package boundary, this project — with a single
concrete processor — has even less reason to.

Option C is rejected because reusing `ISamplerStep` for something that must *also* run in the
greedy path — which structurally bypasses `_steps` today — would misrepresent the contract.
`ISamplerStep.Apply(Span<float>)` means "one step in the probabilistic, temperature-gated chain";
stretching it to also mean "an unconditional hard constraint" collapses the exact distinction this
ADR exists to draw, and leaves a future reader unable to tell, from the interface alone, which
guarantee a given step provides.

### Design

- **`ILogitsProcessor`** — `internal`, in `InferenceEngine.Engine.Sampling`, matching
  `ISamplerStep`'s own visibility (nothing outside `Engine` needs to implement one yet):

  ```csharp
  internal interface ILogitsProcessor
  {
      void Apply(Span<float> logits, ReadOnlySpan<int> generatedTokenIds);
  }
  ```

- **`BannedSequenceLogitsProcessor`** implements it: a length-1 banned sequence is always masked
  to `float.NegativeInfinity`; a longer sequence is masked only when `generatedTokenIds`'s tail
  matches the sequence's prefix — the same algorithm as HuggingFace's
  `NoBadWordsLogitsProcessor`.
- **`BannedSequenceLogitsProcessor.FromWords(ITokenizer, IReadOnlyList<string>)`** — a factory that
  tokenizes literal ban strings once, at session-configuration time, rather than per generated
  token. Takes `Core.ITokenizer` by interface, not the `Tokenizers` project — `Engine` already
  references `Tokenizers` and holds a live tokenizer instance for encode/decode, so this adds no
  new project dependency, only a new use of one that already exists.
- **`SamplingPipeline.Sample`** gains a `ReadOnlySpan<int> generatedTokenIds` parameter and runs
  every registered `ILogitsProcessor` unconditionally, before the greedy/stochastic branch
  splits. This is the one behavioral change this whole ADR exists to make correct — today a ban
  could only ever be wired into the stochastic path, which is not the default.
- **External configuration:** `GenerationOptions` gains `IReadOnlyList<string>? BannedWords`, and
  a new CLI flag/env var, comma-separated (`--ban-words "EPAM,EPAM ,E"` /
  `INFERENCE_BAN_WORDS`) — consistent with the existing scalar-flag pattern (`--temperature`,
  `--top-k`), no new repeated-flag parsing machinery needed.
- **Named seams, explicitly NOT built now:** repetition penalty, forced-token bias, and
  grammar/schema-constrained decoding (dotLLM's `IDecodingConstraint` is the reference shape if
  this is ever wanted — see `docs/investigation/dotllm-architecture-trace.md`). Each is deferred
  to its own future ADR when it has a real caller, per AGENTS.md's "no speculative abstractions" —
  named here so a future reader can see they were anticipated, not discovered as an afterthought.
- **`PrefillTopLogits`/`--debug-logits` stays unprocessed**, deliberately — see Context.

### Consequences

Legend: 🟢 upside · 🟡 accepted trade-off · 🔴 downside.

- 🟢 Banning now works correctly under greedy decoding — the default and, currently, the only path
  a ban could not previously reach at all.
- 🟢 The `ILogitsProcessor`/`ISamplerStep` split makes an implicit distinction (deterministic hard
  constraint vs. probabilistic distribution-shaping) explicit and named, matching how every
  surveyed engine actually organizes this code.
- 🟡 Banning is exact-literal-string and case-sensitive in v1 — the caller supplies variants
  explicitly (e.g. `"EPAM,EPAM ,E"`), rather than the pipeline automatically deriving
  leading-space or case variants. This matches the concrete example that motivated this ADR, and
  keeps `FromWords` simple; automatic variant derivation is not built and not currently planned.
- 🟡 A logits processor sees only the tokens generated *this turn* (`generatedTokenIds`), not the
  prompt — a banned phrase already present in the prompt text is not retroactively scrubbed from
  what the model attends to or from anything already emitted. This is a scope boundary, not a bug:
  the mechanism constrains what gets *generated*, not what the prompt already contains.
- 🟡 `Engine` keeps growing rather than gaining a sibling project. This is an accepted trade-off,
  not a cost-free win — it matches every reference engine's own choice (see the survey above), but
  it does mean `Engine`'s surface area keeps increasing without a project boundary to slow it down.
  If logits processing grows well past banning (multiple processor types, per-processor
  configuration, a plugin registry), that growth — not this ADR's initial scope — would be the
  trigger to revisit Option A.
- 🔴 `GenerationOptions` is a positional record consumed positionally at
  `src/InferenceEngine.Cli/Program.cs:53` (`new GenerationOptions(options.MaxTokens,
  options.Temperature, options.TopK, options.TopP, options.Seed, options.Raw)`). Adding
  `BannedWords` must append it at the end of the parameter list, or that call site breaks
  silently rather than with a compile error, since positional record construction doesn't name
  its arguments. This is a known, small, separate fragility — noted here, fixed as part of
  implementation, not by this ADR.

## Pros and Cons of the Options

### Option A — New project `InferenceEngine.LogitsProcessing`

- 🟢 Mirrors the precedent already set by `Tokenizers` getting its own project for a bounded,
  well-defined concern.
- 🟢 Would give logits processing room to grow (multiple processor types, its own tests project)
  without further crowding `Engine`.
- 🔴 The ceremony (new `.csproj`, solution-file GUID and config rows, cross-reference updates in
  four separate documents) is disproportionate to what's actually shipping this round: one
  interface, one implementation, one factory method.
- 🔴 Not one of the four engines surveyed treats this as a package/project boundary — this option
  would be inventing a boundary none of our reference architectures use, for content sized well
  below what any of them has.
- 🔴 Contradicts [260811](260811-solution-and-project-layout.md)'s own stated caution against
  adding a project ahead of real content ("this ADR doesn't pre-approve that; it'll need its own
  decision when it has real content") — exactly the situation here.

### Option B — Explicit internal split inside `Engine` (chosen)

- 🟢 Matches every surveyed engine's actual design (dotLLM, HuggingFace transformers, vLLM,
  llama.cpp) — a phase split within one pipeline, not a package boundary.
- 🟢 Gives banning (and future deterministic processors) the unconditional-execution guarantee it
  needs, via a signature (`ILogitsProcessor`) that's honest about being a different contract than
  `ISamplerStep`, without any new project ceremony.
- 🟢 Cheap to revisit later: if this does grow enough content to justify Option A, the code already
  lives in its own subfolder/namespace and would be a near-mechanical extraction, not a rewrite.
- 🟡 `Engine` continues to be the project that absorbs orchestration-adjacent growth — same
  trade-off every reference engine has made, not unique to this codebase.

### Option C — No new abstraction, stretch `ISamplerStep`

- 🟢 Zero new interfaces — the smallest possible diff to the type system.
- 🔴 Requires special-casing the greedy path to also run "must-always-run" steps, which is exactly
  the branch this ADR exists to fix cleanly rather than patch around.
- 🔴 Blurs `ISamplerStep`'s meaning: some steps would be probabilistic and temperature-gated,
  others deterministic and unconditional, indistinguishable by type — a future reader (or
  implementer of a new step) would have no way to tell which guarantee applies without reading
  every call site.
- 🔴 Doesn't match any engine surveyed — even the ones that keep logits processing and sampling in
  one module still give them distinct types (`LogitsProcessor` vs. sampler steps in HuggingFace
  and vLLM; a separate "run logit processors" phase vs. "run sampler steps" in dotLLM).

## More Information

- [260811-solution-and-project-layout.md](260811-solution-and-project-layout.md) — the project
  boundaries this ADR operates inside, and the "no project without real content" stance Option A
  is weighed against.
- [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md) —
  put "Sampling / logits" in `Engine`'s build column; this ADR is that row's follow-up.
- [dotllm-architecture-trace.md](../investigation/dotllm-architecture-trace.md) —
  `SamplerPipeline.Sample()`'s logit-processors-then-sampler-steps phase split, our primary
  reference's actual design.
- dotLLM's own documentation (blog posts on logits/logprobs/temperature, linked from README.md).
- [HuggingFace transformers — `LogitsProcessor`/`NoBadWordsLogitsProcessor`](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.NoBadWordsLogitsProcessor) —
  the sequence-prefix-matching algorithm this ADR's `BannedSequenceLogitsProcessor` follows.
- [vLLM — Logits Processors design doc](https://docs.vllm.ai/en/latest/design/logits_processors.html) —
  source for the "hard-coded into the sampler... not yet utilizing the programming model"
  architectural-debt admission cited above.
- [llama.cpp — `llama_sampler_chain` / `logit_bias`](https://github.com/ggml-org/llama.cpp) —
  the single-chain design where bias is the chain's first link, not a separate module.
- **Known, separate, out-of-scope gap:** `docs/architecture/planned-tokenization.md`'s stated
  stance ("reuse a tokenizer library") contradicts the actual code (`GgufBpeTokenizer`, hand-rolled
  byte-level BPE, zero NuGet dependencies), and no ADR records that reversal. Not addressed here —
  flagged for a future round when that placeholder is filled in.

## Decision Log

| Date       | Change            | By                 |
|------------|-------------------|--------------------|
| 2026-09-17 | Initial proposal  | Aleksei Kolesnikov |
