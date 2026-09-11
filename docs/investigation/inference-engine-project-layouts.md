# Inference-Engine Project Layouts (Reference Comparison)

How other LLM-inference engines structure their codebases, gathered to justify our own
5-project split (see [the solution-layout ADR](../architecture/260811-solution-and-project-layout.md)).
Two axes matter for us: **build vs. reuse** (hand-rolled kernels vs. wrapping a native runtime)
and **structural granularity** (single file → many modules).

## Summary table

| Project | Lang | Build vs. reuse | Structure (top-level modules) | Compute | Server layer | Model format |
|---------|------|-----------------|-------------------------------|---------|--------------|--------------|
| **dotLLM** (our primary reference) | C#/.NET | Ground-up | ~17 projects | Custom SIMD (AVX2/512) + PTX CUDA | Yes (OpenAI-compatible) | GGUF |
| **Jlama** | Java 20+ | Ground-up | ~6 Maven modules: `jlama-core`, `jlama-native`, `jlama-cli`, `jlama-net`, `jlama-tests`, k8s | Custom kernels (Java Vector API "Panama", optional native SIMD, experimental WebGPU) | Yes (OpenAI-compatible REST `restapi`; distributed coordinator/worker) | HuggingFace SafeTensors (Q8/Q4) — **not GGUF** |
| **llama3.java** | Java 21+ | Ground-up | **Single file** (`Llama3.java`, no dependencies) | Custom SIMD via Java Vector API | None in core (external Spring Boot wrapper exists) | GGUF (own parser; Q4_0…Q8_0, F16/BF16/F32) |
| **LLamaSharp** | C#/.NET | Wrapper over llama.cpp (git submodule) | 7 projects: `LLama` (core), `.Examples`, `.Unittest`, `.Web`, `.WebAPI`, `.Benchmark`, `.Mobile` | Reuses native llama.cpp backends (CPU/CUDA/Vulkan/Metal as separate NuGet pkgs) | Partial (`LLama.WebAPI`, `LLama.Web` demo; community LLamaWorker) | GGUF |
| **ONNX Runtime GenAI** | C/C++ core + C#/Py/Java/ObjC bindings | Orchestration layer atop ONNX Runtime | Directory-based SDK: `/src`, `/examples`, `/docs`, `/test`, `/tools`, `/cmake` | Reuses ONNX Runtime kernels | None (SDK library) | ONNX only (convert from HF first) |
| **LM-Kit.NET** | C#/.NET (commercial) | Wrapper over llama.cpp backend | Commercial NuGet SDK (module split unverified) | Reuses llama.cpp backend | Unverified | GGUF primarily; also ONNX + proprietary LMK |

## Synthesis

- **Build vs. reuse splits by ecosystem.** The pure-Java engines (Jlama, llama3.java) are both
  ground-up — they hand-roll kernels using Java's Vector API because there's no ubiquitous native
  runtime they'd default to. The C# engines (LLamaSharp, ONNX Runtime GenAI, LM-Kit.NET) almost
  all *wrap* a native runtime (llama.cpp or ONNX Runtime) instead of writing kernels. **dotLLM is
  the C# outlier: ground-up, custom SIMD + CUDA.**
- **Granularity spans the full range.** llama3.java is the extreme monolith (one file, no deps);
  Jlama shows a from-scratch engine factored into ~6 modules; LLamaSharp's 7 projects are split
  largely by *deployment target* (web/API/mobile/benchmark) rather than by inference concern;
  dotLLM's ~17 are split by inference concern *and* backend *and* serving.
- **Where our 5-project split sits.** Far more layered than llama3.java's single file, comparable
  in count to Jlama's real modules, and much lighter than dotLLM or LLamaSharp — with boundaries
  drawn by **inference concern** (Core/Models/Tokenizers/Engine/Cli), not by deployment target.

## Corrections / caveats captured during research

- **Jlama consumes SafeTensors, not GGUF** — an intermediate search claimed GGUF; Jlama's own
  README lists only HuggingFace SafeTensors. GGUF-in-Java is **llama3.java**.
- **LM-Kit.NET server layer**: unverified from the sources checked — not treated as confirmed.
- **LM-Kit.NET module structure**: commercial SDK, internal project split not public.

## Sources

- dotLLM — https://github.com/kkokosa/dotLLM (also `dotllm-architecture-trace.md`)
- Jlama — https://github.com/tjake/Jlama
- llama3.java — https://github.com/mukel/llama3.java
- LLamaSharp — https://github.com/SciSharp/LLamaSharp
- ONNX Runtime GenAI — https://github.com/microsoft/onnxruntime-genai
- LM-Kit.NET — https://docs.lm-kit.com/lm-kit-net/guides/faq/use-custom-gguf-models.html
