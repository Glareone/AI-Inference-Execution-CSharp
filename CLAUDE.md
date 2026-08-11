# CLAUDE.md

This project's ground rules (stack, `unsafe` policy, ADR process, working style) live in
[AGENTS.md](AGENTS.md), written to be portable across AI coding assistants — read that first.

## Claude Code specifics

- Subagents live in [.claude/agents/](.claude/agents/):
  - `adr-writer-reviewer` — write or review ADRs in `architecture/` (MADR format). Use this
    instead of writing ADRs freehand.
  - `csharp-dotnet` — C#/.NET implementation and review, enforcing the stack and `unsafe`
    policy from AGENTS.md.
- No project-specific skills or hooks yet; add them under `.claude/skills/` and configure
  hooks in `.claude/settings.json` as real, recurring needs show up — don't scaffold empty
  ones speculatively.
