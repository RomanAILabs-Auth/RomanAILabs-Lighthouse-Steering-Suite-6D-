# Lighthouse-FCES + WhiteHole Steering + RAIL-FPMF (6D) — Single-File LLM Steering Runner
Copyright Daniel Harding - RomanAILabs  
Credits: OpenAI GPT-5.2 Thinking

> **What this is:** a **runtime control layer** for GGUF models via **llama.cpp** (`llama-cpp-python`) that steers sampling **per token** using a compact signal stack:
> - **Lighthouse FCES** (Field–Curvature Entanglement Scalar)
> - **WhiteHole Steering** (bounded heavy-tailed scalar)
> - **RAIL-FPMF 6D controller** (dynamic strength + optional EOS nudge)
>
> **What it is NOT:** training, finetuning, RLHF, or a new model. This is a **sampling-time controller**.

---

## Why you’d use this
If you’re tired of:
- looping / repetitive generations
- “stuck” low-entropy degeneracy
- brittle sampling behavior across prompts

…this script gives you a **measurable**, **bounded**, **CPU-friendly** way to push the sampler back toward healthier dynamics.

It also includes a benchmark mode that measures **TRUE overhead** by forcing **EXACT token counts** (EOS suppression), so you can compare baseline vs steered honestly.

---

## Features
- ✅ Single file. Drop it anywhere. Run it.
- ✅ Works with **GGUF + llama.cpp** through `llama-cpp-python`
- ✅ `run` mode (one generation)
- ✅ `bench` mode (baseline vs steered over trials)
- ✅ **EXACT token bench** (bans EOS to generate exactly `n_predict` tokens)
- ✅ Lightweight quality metrics (distinct-1/2, repeat rate) to sanity-check changes
- ✅ Guardrails: clipping, stable sigmoid/softmax, safe fallbacks

---

## Install
```bash
pip install llama-cpp-python numpy
