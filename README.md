# AI-Inference-Execution-CSharp

An investigation into how LLM inference engines work, implemented in C#/.NET.

## Inspiration

This project was sparked by Konrad Kokosa's [dotLLM](https://github.com/kkokosa/dotLLM) — a
from-scratch LLM inference engine written natively in .NET (GGUF loading, tokenization,
attention, sampling, SIMD CPU kernels, CUDA via the driver API, an OpenAI-compatible server).
His [write-up](https://kokosa.dev/blog/2026/dotllm/) on building it — and, more importantly,
on using AI-assisted development with real discipline (written design docs, staged plans,
multi-model review) to get there — is the direct inspiration for both *what* this project
explores and *how* it's built.

dotLLM proved a full-scale, high-performance engine is possible in C#. This project is not
an attempt to repeat that at the same scale. It's a smaller, guided investigation: understand
how the pieces of an inference engine fit together (model loading, tokenization, attention,
sampling, KV-cache, serving) by building a working — not necessarily fast — version, leaning
on existing .NET libraries where that lets us focus on the architecture and the engine layer
rather than reimplementing math kernels from scratch.

## Goals

- Understand how an LLM inference engine is structured, end to end, well enough to explain it.
- Reuse mature libraries (tokenizers, tensor/math, model format parsing) instead of
  hand-rolling kernels — hand-rolled/`unsafe` code is the exception, not the default, and
  requires explicit justification (see [AGENTS.md](AGENTS.md)).
- Use this repo as a proving ground for an agentic coding workflow: architecture decisions
  are captured as ADRs, an AI code-review pass happens before anything lands, and structure
  (docs, plans) drives the AI rather than the other way around — the same lesson Konrad's
  write-up lands on: *"AI amplifies discipline; it doesn't replace it."*

## Status

Early scaffolding: solution/project layout exists (see
[ADR-0001](architecture/0001-solution-and-project-layout.md)) with a runnable but empty CLI.
No model loading, tokenization, or generation yet — see [CHANGELOG.md](CHANGELOG.md) for
progress.

## Project docs

- [AGENTS.md](AGENTS.md) — ground rules for any AI coding assistant working in this repo
  (stack, `unsafe` policy, ADR process). [CLAUDE.md](CLAUDE.md) points here and adds
  Claude Code–specific notes (subagents, etc.).
- [architecture/](architecture/) — architecture decisions, in [MADR](https://adr.github.io/madr/) format.
- [CHANGELOG.md](CHANGELOG.md) — notable changes, [Keep a Changelog](https://keepachangelog.com/) style.
