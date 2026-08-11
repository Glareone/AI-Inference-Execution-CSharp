# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffolding: README with project goal and inspiration (dotLLM by Konrad Kokosa),
  AGENTS.md / CLAUDE.md ground rules, ADR template (`architecture/`), and Claude Code
  subagents (`adr-writer-reviewer`, `csharp-dotnet`).
- [ADR-0001](architecture/0001-solution-and-project-layout.md): solution/project layout.
- `src/` scaffolding per ADR-0001 — `InferenceEngine.{Core,Models,Tokenizers,Engine}` class
  libraries and a runnable `InferenceEngine.Cli` console app (net10.0, no functionality yet).
- Rider run/debug configuration for `InferenceEngine.Cli` (`launchSettings.json` +
  `.idea/runConfigurations/`).
- `adr-writer-reviewer` and `csharp-dotnet` agents now consult context7 for current library
  docs before evaluating/using a package, instead of relying on training-data memory.
