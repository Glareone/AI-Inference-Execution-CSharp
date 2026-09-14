# CLAUDE.md

This project's ground rules (stack, `unsafe` policy, ADR process, working style) live in
[AGENTS.md](AGENTS.md), written to be portable across AI coding assistants — read that first.

## Claude Code specifics

- Subagents live in [.claude/agents/](.claude/agents/):
  - `adr-author` — write or review ADRs in `docs/architecture/` (MADR format). Use this instead
    of writing ADRs freehand. Every ADR it writes or touches must end with a filled-in
    "Decision Log" table.
  - `csharp-dotnet` — C#/.NET implementation and review, enforcing the stack and `unsafe`
    policy from AGENTS.md.
  - `test-writer-runner` — write, update, and run tests for any `InferenceEngine.*` project
    (one test project per `src/` project, business-case-documented). Use after implementing or
    changing behavior; always runs `dotnet test` and confirms it passes.
  - `researcher` — web research on inference internals (algorithms, specs, papers). Use
    before making implementation decisions.
  - `huggingface-explorer` — HuggingFace ecosystem (model formats, APIs, downloads).
  - `code-reader` — read and analyze external repos (dotLLM, LLamaSharp) to understand
    architecture and patterns. Not for writing code in our project.
- All agents use context7 MCP for current library documentation.
- No project-specific skills or hooks yet; add them under `.claude/skills/` and configure
  hooks in `.claude/settings.json` as real, recurring needs show up — don't scaffold empty
  ones speculatively.
