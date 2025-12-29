#!/usr/bin/env python3
# Copyright Daniel Harding - RomanAILabs
# Credits: OpenAI GPT-5.2 Thinking
"""
Lighthouse-FCES + WhiteHole Steering + RAIL-FPMF (6D)
=====================================================

Single-file, GitHub-ready reference implementation.

What this is
------------
A *runtime* steering / control layer for GGUF models running via llama.cpp (llama-cpp-python).
It implements three cooperating parts:

1) Lighthouse FCES (Field–Curvature Entanglement Scalar)
   - A compact scalar computed online from model sampling signals (repetition, entropy, top-token share, and their curvature).
   - Interpretable: higher FCES ≈ "field/curvature entanglement" (non-separable dynamics) → steer harder.

2) WhiteHole Steering
   - A bounded, heavy-tailed steering scalar s in [-1, 1] computed from (x, y) where:
       x = a 6D "spacetime coordinate" (token progress or constant),
       y = a Lighthouse quality signal (includes FCES).
   - Uses a weighted-sum sigmoid kernel, then compresses with tanh(sign(S)*ln(1+|S|)).

3) RAIL-FPMF (Flux–Potential–Memory Field, 6D controller)
   - A small 6D state machine that converts FCES + other signals into:
       - logits shaping strength (lam)
       - optional EOS "completion bias" to finish earlier (fewer tokens for same quality)

The script supports:
- run: generate once
- bench: baseline vs steered across trials with timing + lightweight quality metrics
- exact token bench: generates EXACT n_predict tokens (by suppressing EOS)

Install
-------
pip install llama-cpp-python numpy

Examples
--------
# simplest: run (defaults to inserting "run" if you forget the subcommand)
python3 lighthouse_whitehole_fces_fpmf_6d.py --model /path/to/model.gguf --prompt "3 sentences max."

# benchmark
python3 lighthouse_whitehole_fces_fpmf_6d.py bench --model /path/to/model.gguf --prompt "..." --trials 5 --n-predict 96

Notes
-----
- This is NOT training a new model; it's a "control model" that steers sampling.
- Designed for CPU-friendly overhead (single softmax + a few vector ops per token).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    m = np.max(x)
    y = x - m
    np.exp(y, out=y)
    s = float(np.sum(y))
    if s == 0.0 or not np.isfinite(s):
        # fallback uniform
        return np.full_like(x, 1.0 / float(len(x)), dtype=np.float64)
    return (y / s).astype(np.float64, copy=False)

def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(np.float64, copy=False), eps, 1.0)
    h = -float(np.sum(p * np.log(p)))
    return h

def entropy_topk_from_logits(logits: np.ndarray, k: int = 32) -> float:
    # normalized entropy in [0,1] over top-k
    k = int(max(2, min(k, logits.shape[-1])))
    idx = np.argpartition(logits, -k)[-k:]
    top = logits[idx].astype(np.float64, copy=False)
    p = softmax_np(top)
    h = entropy_from_probs(p)
    hmax = math.log(float(k))
    return float(h / hmax) if hmax > 0 else 0.0

def repetition_ratio(tokens: Sequence[int], window: int = 64) -> float:
    if not tokens:
        return 0.0
    w = int(max(1, window))
    tail = tokens[-w:]
    if len(tail) <= 1:
        return 0.0
    uniq = len(set(tail))
    return float(1.0 - (uniq / float(len(tail))))

def top_token_share_from_logits(logits: np.ndarray) -> float:
    p = softmax_np(logits)
    return float(np.max(p))

def distinct_1(text: str) -> float:
    toks = [t for t in re.findall(r"\w+|\S", text) if t.strip()]
    if not toks:
        return 0.0
    return float(len(set(toks)) / len(toks))

def distinct_2(text: str) -> float:
    toks = [t for t in re.findall(r"\w+|\S", text) if t.strip()]
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return float(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

def repeat_rate(text: str) -> float:
    toks = [t for t in re.findall(r"\w+|\S", text) if t.strip()]
    if not toks:
        return 0.0
    return float(1.0 - (len(set(toks)) / len(toks)))

# -----------------------------------------------------------------------------
# 6D math core (tiny + practical)
# -----------------------------------------------------------------------------

@dataclass
class Vec6:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @staticmethod
    def zeros() -> "Vec6":
        return Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __add__(self, o: "Vec6") -> "Vec6":
        return Vec6(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d, self.e + o.e, self.f + o.f)

    def __mul__(self, k: float) -> "Vec6":
        return Vec6(self.a * k, self.b * k, self.c * k, self.d * k, self.e * k, self.f * k)

    def dot(self, o: "Vec6") -> float:
        return (self.a * o.a + self.b * o.b + self.c * o.c + self.d * o.d + self.e * o.e + self.f * o.f)

    def norm(self) -> float:
        return math.sqrt(max(0.0, self.dot(self)))

# -----------------------------------------------------------------------------
# Lighthouse FCES (Field–Curvature Entanglement Scalar) in a computable form
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FCESConfig:
    # "field" weights
    w_rep: float = 1.00
    w_lowent: float = 0.90
    w_topshare: float = 0.60

    # "curvature" weights (deltas)
    w_drep: float = 0.80
    w_dent: float = 0.80
    w_dtop: float = 0.40

    # smoothing / scaling
    eps: float = 1e-9
    fces_gain: float = 1.00  # final multiplier
    fces_clip: float = 2.00  # clip final fces

class LighthouseFCES6D:
    """
    Minimal, online FCES estimator.

    Field vector F (3D)      : [rep, (1-ent), top_share]
    Curvature vector C (3D)  : deltas of the above over time
    Entanglement scalar      : ||F x C|| / (||F||*||C|| + eps) in [0,1]
    Then scaled by magnitudes to become a stronger scalar.
    """
    def __init__(self, cfg: FCESConfig):
        self.cfg = cfg
        self._prev_rep = 0.0
        self._prev_ent = 0.0
        self._prev_top = 0.0
        self._init = False

    def step(self, rep: float, ent: float, top_share: float) -> Tuple[float, Vec6]:
        rep = float(clamp(rep, 0.0, 1.0))
        ent = float(clamp(ent, 0.0, 1.0))
        top_share = float(clamp(top_share, 0.0, 1.0))
        lowent = 1.0 - ent

        if not self._init:
            drep, dent, dtop = 0.0, 0.0, 0.0
            self._init = True
        else:
            drep = rep - self._prev_rep
            dent = ent - self._prev_ent
            dtop = top_share - self._prev_top

        self._prev_rep, self._prev_ent, self._prev_top = rep, ent, top_share

        F = np.array([
            self.cfg.w_rep * rep,
            self.cfg.w_lowent * lowent,
            self.cfg.w_topshare * top_share,
        ], dtype=np.float64)
        C = np.array([
            self.cfg.w_drep * drep,
            self.cfg.w_dent * dent,
            self.cfg.w_dtop * dtop,
        ], dtype=np.float64)

        Fn = float(np.linalg.norm(F))
        Cn = float(np.linalg.norm(C))
        cross = float(np.linalg.norm(np.cross(F, C))) if (Fn > 0 and Cn > 0) else 0.0
        sin_theta = cross / (Fn * Cn + self.cfg.eps) if (Fn > 0 and Cn > 0) else 0.0

        fces = self.cfg.fces_gain * sin_theta * (Fn * Cn)
        fces = float(clamp(fces, -self.cfg.fces_clip, self.cfg.fces_clip))

        v6 = Vec6(rep, lowent, top_share, drep, dent, dtop)
        return fces, v6

# -----------------------------------------------------------------------------
# WhiteHole formula (v12of10-compatible)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WhiteHoleFormulaConfig:
    N: int = 512
    a1: float = 1.0
    alpha: float = 1.35
    beta: float = 0.85
    exp_clip: float = 60.0
    inner_clip: float = 60.0

def white_hole_steering(x: float, y: float, cfg: WhiteHoleFormulaConfig, T: Callable[[float, int], float]) -> float:
    """
    s = tanh( sign(S) * ln(1+|S|) )
    where
      S = (Σ a1 f(i) σ(f(i)K(i)T(x,i)T(y,0)exp(-iT(x,i)T(y,0)))) / (Σ f(i))
      f(i)=(i+1)^(-alpha), K(i)=(i+1)^(-beta), σ=sigmoid
    """
    Ty0 = float(T(y, 0))
    numerator = 0.0
    denom = 0.0
    for i in range(1, int(cfg.N) + 1):
        fi = (i + 1) ** (-cfg.alpha)
        Ki = (i + 1) ** (-cfg.beta)
        Txi = float(T(x, i))

        expo_arg = clamp(-i * Txi * Ty0, -cfg.exp_clip, cfg.exp_clip)
        exp_term = math.exp(expo_arg)

        inner = fi * Ki * Txi * Ty0 * exp_term
        inner = clamp(inner, -cfg.inner_clip, cfg.inner_clip)

        g = sigmoid(inner)

        numerator += cfg.a1 * fi * g
        denom += fi

    S = numerator / denom if denom != 0.0 else 0.0
    phi = math.copysign(math.log1p(abs(S)), S)
    return math.tanh(phi)

# -----------------------------------------------------------------------------
# RAIL-FPMF: 6D controller (Flux–Potential–Memory Field)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FPMFConfig:
    enabled: bool = True

    # core control
    base_strength: float = 2.00
    strength_gain: float = 1.15
    strength_clip: float = 6.00

    # temperature control
    base_temp: float = 0.75
    temp_span: float = 0.25
    temp_min: float = 0.50
    temp_max: float = 1.35

    # EOS / brevity
    enable_eos_nudge: bool = True
    eos_nudge: float = 1.25
    eos_nudge_clip: float = 5.0
    min_tokens_before_eos: int = 24
    done_entropy_threshold: float = 0.45
    done_punct_regex: str = r"[.!?…]\s*$"

    # state dynamics
    dt: float = 0.25
    decay: float = 0.92

class RAILFPMF6D:
    """
    Small 6D dynamical controller. Inputs: Vec6(rep, lowent, top, drep, dent, dtop).
    """
    def __init__(self, cfg: FPMFConfig):
        self.cfg = cfg
        self.state = Vec6.zeros()
        self._punct_re = re.compile(cfg.done_punct_regex)

    def reset(self) -> None:
        self.state = Vec6.zeros()

    def step(self, v: Vec6) -> Vec6:
        dt = float(self.cfg.dt)
        d = float(self.cfg.decay)
        inc = Vec6(v.a, v.b, v.c, v.d, -v.e, v.f)
        self.state = (self.state * d) + (inc * dt)
        return self.state

    def strength(self, s_wh: float) -> float:
        if not self.cfg.enabled:
            return 0.0
        mag = self.state.norm()
        lam = self.cfg.base_strength * (1.0 + self.cfg.strength_gain * mag) * (1.0 + 0.25 * abs(float(s_wh)))
        return float(clamp(lam, 0.0, self.cfg.strength_clip))

    def temperature(self, s_wh: float) -> float:
        t = self.cfg.base_temp + (self.cfg.temp_span * float(s_wh))
        return float(clamp(t, self.cfg.temp_min, self.cfg.temp_max))

    def eos_bias(self, token_i: int, ent: float, text_so_far: str) -> float:
        if not (self.cfg.enabled and self.cfg.enable_eos_nudge):
            return 0.0
        if token_i < int(self.cfg.min_tokens_before_eos):
            return 0.0
        if float(ent) > float(self.cfg.done_entropy_threshold):
            return 0.0
        if not self._punct_re.search(text_so_far):
            return 0.0
        return float(clamp(self.cfg.eos_nudge, 0.0, self.cfg.eos_nudge_clip))

# -----------------------------------------------------------------------------
# Steering config (ties everything together)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SteeringConfig6D:
    enabled: bool = True

    # quality y composition
    y_bias: float = -0.10
    w_rep: float = 1.20
    w_lowent: float = 0.90
    w_fces: float = 1.15

    # whitehole x
    x_mode: str = "progress"   # "progress" or "const"
    x_const: float = 0.70

    # logits shaping
    logits_shape_clip: float = 10.0
    rep_strength_boost: float = 1.25

    # entropy/repetition windows
    rep_window: int = 64
    ent_topk: int = 32

    # exact-token behavior
    exact_mode_ban_eos: bool = True
    eos_ban_logit: float = -100.0

# -----------------------------------------------------------------------------
# llama.cpp integration (llama-cpp-python)
# -----------------------------------------------------------------------------

def _lazy_import_llama():
    try:
        from llama_cpp import Llama, LogitsProcessorList
        return Llama, LogitsProcessorList
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: llama-cpp-python. Install with:\n"
            "  pip install llama-cpp-python\n"
            f"Original error: {e}"
        )

def find_gguf_in_dir(folder: str, pick_regex: Optional[str] = None) -> Optional[str]:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        return None
    ggufs = sorted(
        [p for p in folder_path.glob("*.gguf") if p.is_file()],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not ggufs:
        return None
    if pick_regex:
        r = re.compile(pick_regex, re.IGNORECASE)
        for p in ggufs:
            if r.search(p.name):
                return str(p)
    return str(ggufs[0])

def resolve_model_path(model: Optional[str], model_dir: Optional[str], pick_regex: Optional[str]) -> str:
    if model:
        p = Path(model).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return str(p)
    search_dir = model_dir or os.getcwd()
    found = find_gguf_in_dir(search_dir, pick_regex=pick_regex)
    if not found:
        raise FileNotFoundError(
            f"No .gguf found in: {search_dir}\n"
            "Put your model in the same folder as this script, or use --model-dir / --model"
        )
    return found

# -----------------------------------------------------------------------------
# Generation loop (EXACT tokens; supports logits processors)
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    text: str
    elapsed_s: float
    tok_s: float
    n_tokens: int
    hook_calls: int
    avg_s: float
    avg_y: float
    avg_fces: float

def make_logits_processor_6d(
    llm,
    n_predict: int,
    fcfg: WhiteHoleFormulaConfig,
    scfg: SteeringConfig6D,
    fces: LighthouseFCES6D,
    fpmf: RAILFPMF6D,
    *,
    steer_enabled: bool,
    exact_mode: bool,
) -> Tuple[object, Dict[str, float]]:
    stats: Dict[str, float] = {
        "hook_calls": 0.0,
        "sum_s": 0.0,
        "sum_y": 0.0,
        "sum_fces": 0.0,
    }

    try:
        eos_id = int(llm.token_eos())
    except Exception:
        eos_id = -1

    detok_buf: List[int] = []

    def processor(input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        stats["hook_calls"] += 1.0

        out = scores.astype(np.float64, copy=True)

        if not steer_enabled:
            if exact_mode and scfg.exact_mode_ban_eos and eos_id >= 0:
                out[eos_id] = float(out[eos_id] + scfg.eos_ban_logit)
            return out.astype(np.float32, copy=False)

        history = input_ids.tolist()
        rep = repetition_ratio(history, window=scfg.rep_window)
        ent = entropy_topk_from_logits(scores, k=scfg.ent_topk)
        top_share = top_token_share_from_logits(scores)

        fces_val, v6 = fces.step(rep=rep, ent=ent, top_share=top_share)
        _ = fpmf.step(v6)

        y = (
            scfg.y_bias
            + scfg.w_rep * rep
            + scfg.w_lowent * (1.0 - ent)
            + scfg.w_fces * fces_val
        )

        if scfg.x_mode.lower() == "progress":
            x = float(len(history) / max(1, int(n_predict)))
        else:
            x = float(scfg.x_const)

        def T_func(val: float, i: int) -> float:
            return float(val) * float(i + 1)

        s = float(white_hole_steering(x=x, y=y, cfg=fcfg, T=T_func))
        s = float(clamp(s, -1.0, 1.0))

        lam = fpmf.strength(s_wh=s)
        rep_boost = 1.0 + (rep * max(0.0, scfg.rep_strength_boost - 1.0))
        lam *= float(rep_boost)

        probs = softmax_np(scores)
        delta = lam * s * probs
        delta = np.clip(delta, -scfg.logits_shape_clip, scfg.logits_shape_clip)
        out = out - delta

        if exact_mode and scfg.exact_mode_ban_eos and eos_id >= 0:
            out[eos_id] = float(out[eos_id] + scfg.eos_ban_logit)
        elif (not exact_mode) and eos_id >= 0:
            detok_buf.append(int(history[-1]) if history else 0)
            tail = detok_buf[-128:]
            try:
                text_tail = llm.detokenize(tail).decode("utf-8", errors="ignore")
            except Exception:
                text_tail = ""
            eb = fpmf.eos_bias(token_i=len(history), ent=ent, text_so_far=text_tail)
            if eb != 0.0:
                out[eos_id] = float(out[eos_id] + clamp(eb, -10.0, 10.0))

        stats["sum_s"] += abs(s)
        stats["sum_y"] += float(y)
        stats["sum_fces"] += float(fces_val)

        return out.astype(np.float32, copy=False)

    return processor, stats

def generate_exact(
    llm,
    prompt: str,
    n_predict: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    typical_p: float,
    repeat_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    tfs_z: float,
    mirostat_mode: int,
    mirostat_tau: float,
    mirostat_eta: float,
    penalize_nl: bool,
    logits_processor_list,
) -> Tuple[List[int], float]:
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.reset()
    llm.eval(prompt_tokens)
    out_tokens: List[int] = []
    t0 = _now()

    for _ in range(int(n_predict)):
        tok = llm.sample(
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temperature,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            penalize_nl=penalize_nl,
            logits_processor=logits_processor_list,
        )
        out_tokens.append(int(tok))
        llm.eval([int(tok)])

    t1 = _now()
    return out_tokens, (t1 - t0)

def detokenize(llm, tokens: Sequence[int]) -> str:
    try:
        return llm.detokenize(list(tokens)).decode("utf-8", errors="ignore")
    except Exception:
        return "".join([str(t) for t in tokens])

def run_once(
    model_path: str,
    prompt: str,
    n_ctx: int,
    n_predict: int,
    threads: int,
    gpu_layers: int,
    seed: int,
    *,
    steer: bool,
    exact_tokens: bool,
    # sampling
    top_k: int,
    top_p: float,
    min_p: float,
    typical_p: float,
    temperature: float,
    repeat_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    tfs_z: float,
    mirostat_mode: int,
    mirostat_tau: float,
    mirostat_eta: float,
    penalize_nl: bool,
    # configs
    fcfg: WhiteHoleFormulaConfig,
    scfg: SteeringConfig6D,
    fces_cfg: FCESConfig,
    fpmf_cfg: FPMFConfig,
) -> RunResult:
    Llama, LogitsProcessorList = _lazy_import_llama()

    llm = Llama(
        model_path=model_path,
        n_ctx=int(n_ctx),
        n_threads=int(threads),
        n_gpu_layers=int(gpu_layers),
        seed=int(seed) if seed is not None else 0,
        logits_all=True,
        verbose=False,
    )

    fces = LighthouseFCES6D(fces_cfg)
    fpmf = RAILFPMF6D(fpmf_cfg)

    processor, stats = make_logits_processor_6d(
        llm=llm,
        n_predict=int(n_predict),
        fcfg=fcfg,
        scfg=scfg,
        fces=fces,
        fpmf=fpmf,
        steer_enabled=bool(steer and scfg.enabled),
        exact_mode=bool(exact_tokens),
    )
    lp = LogitsProcessorList([processor])

    toks, elapsed = generate_exact(
        llm=llm,
        prompt=prompt,
        n_predict=int(n_predict),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        typical_p=float(typical_p),
        repeat_penalty=float(repeat_penalty),
        frequency_penalty=float(frequency_penalty),
        presence_penalty=float(presence_penalty),
        tfs_z=float(tfs_z),
        mirostat_mode=int(mirostat_mode),
        mirostat_tau=float(mirostat_tau),
        mirostat_eta=float(mirostat_eta),
        penalize_nl=bool(penalize_nl),
        logits_processor_list=lp,
    )

    txt = detokenize(llm, toks)
    n = len(toks)
    tok_s = (n / elapsed) if elapsed > 0 else 0.0

    hook_calls = int(stats["hook_calls"])
    denom = max(1, hook_calls)
    avg_s = float(stats["sum_s"] / denom)
    avg_y = float(stats["sum_y"] / denom)
    avg_fces = float(stats["sum_fces"] / denom)

    return RunResult(
        text=txt,
        elapsed_s=float(elapsed),
        tok_s=float(tok_s),
        n_tokens=n,
        hook_calls=hook_calls,
        avg_s=avg_s,
        avg_y=avg_y,
        avg_fces=avg_fces,
    )

# -----------------------------------------------------------------------------
# Benchmark + reporting (guild-friendly)
# -----------------------------------------------------------------------------

def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0

def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92 + "\n")

def bench(args, model_path: str) -> None:
    prompt = args.prompt
    trials = int(args.trials)
    n_predict = int(args.n_predict)

    fcfg = WhiteHoleFormulaConfig()
    scfg = SteeringConfig6D()
    fces_cfg = FCESConfig()
    fpmf_cfg = FPMFConfig()

    print_header("2) Warmup")
    _ = run_once(
        model_path=model_path,
        prompt="Warmup. Say OK.",
        n_ctx=args.n_ctx,
        n_predict=min(16, n_predict),
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        seed=args.seed,
        steer=False,
        exact_tokens=True,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        typical_p=args.typical_p,
        temperature=args.temp,
        repeat_penalty=args.repeat_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        tfs_z=args.tfs_z,
        mirostat_mode=args.mirostat_mode,
        mirostat_tau=args.mirostat_tau,
        mirostat_eta=args.mirostat_eta,
        penalize_nl=not args.no_penalize_nl,
        fcfg=fcfg,
        scfg=scfg,
        fces_cfg=fces_cfg,
        fpmf_cfg=fpmf_cfg,
    )
    print("[RomanAILabs] Warmup complete.\n")

    print_header("3) Trials (baseline vs steered 6D, EXACT tokens)")

    base_times: List[float] = []
    steer_times: List[float] = []
    base_tps: List[float] = []
    steer_tps: List[float] = []
    base_texts: List[str] = []
    steer_texts: List[str] = []
    steer_avg_s: List[float] = []
    steer_avg_fces: List[float] = []

    for t in range(1, trials + 1):
        base = run_once(
            model_path=model_path,
            prompt=prompt,
            n_ctx=args.n_ctx,
            n_predict=n_predict,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
            seed=args.seed + t if args.seed is not None else 0,
            steer=False,
            exact_tokens=True,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            typical_p=args.typical_p,
            temperature=args.temp,
            repeat_penalty=args.repeat_penalty,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            tfs_z=args.tfs_z,
            mirostat_mode=args.mirostat_mode,
            mirostat_tau=args.mirostat_tau,
            mirostat_eta=args.mirostat_eta,
            penalize_nl=not args.no_penalize_nl,
            fcfg=fcfg,
            scfg=scfg,
            fces_cfg=fces_cfg,
            fpmf_cfg=fpmf_cfg,
        )
        steered = run_once(
            model_path=model_path,
            prompt=prompt,
            n_ctx=args.n_ctx,
            n_predict=n_predict,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
            seed=(args.seed + 1000 + t) if args.seed is not None else 0,
            steer=True,
            exact_tokens=True,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            typical_p=args.typical_p,
            temperature=args.temp,
            repeat_penalty=args.repeat_penalty,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            tfs_z=args.tfs_z,
            mirostat_mode=args.mirostat_mode,
            mirostat_tau=args.mirostat_tau,
            mirostat_eta=args.mirostat_eta,
            penalize_nl=not args.no_penalize_nl,
            fcfg=fcfg,
            scfg=scfg,
            fces_cfg=fces_cfg,
            fpmf_cfg=fpmf_cfg,
        )

        base_times.append(base.elapsed_s)
        steer_times.append(steered.elapsed_s)
        base_tps.append(base.tok_s)
        steer_tps.append(steered.tok_s)
        base_texts.append(base.text)
        steer_texts.append(steered.text)
        steer_avg_s.append(steered.avg_s)
        steer_avg_fces.append(steered.avg_fces)

        overhead = pct_change(steered.elapsed_s, base.elapsed_s)
        tps_delta = pct_change(steered.tok_s, base.tok_s)

        print(
            f"[RomanAILabs] Trial {t:02d}/{trials} | "
            f"base={base.elapsed_s:.3f}s ({base.tok_s:.2f} tok/s) | "
            f"steer6D={steered.elapsed_s:.3f}s ({steered.tok_s:.2f} tok/s) | "
            f"TRUE overhead={overhead:+.2f}% | TPS change={tps_delta:+.2f}% | "
            f"avg_s={steered.avg_s:.4f} avg_fces={steered.avg_fces:.4f} | hook_calls={steered.hook_calls}"
        )

        if args.show_excerpts and t == 1:
            print("\n[RomanAILabs] Excerpt baseline:\n---\n" + base.text[:320] + "\n---")
            print("\n[RomanAILabs] Excerpt steered (6D):\n---\n" + steered.text[:320] + "\n---\n")

    print_header("4) Summary (averages)")
    b_avg = float(np.mean(base_times))
    s_avg = float(np.mean(steer_times))
    b_tps = float(np.mean(base_tps))
    s_tps = float(np.mean(steer_tps))

    overhead_avg = pct_change(s_avg, b_avg)
    tps_delta_avg = pct_change(s_tps, b_tps)

    d1_base = float(np.mean([distinct_1(x) for x in base_texts]))
    d1_steer = float(np.mean([distinct_1(x) for x in steer_texts]))
    d2_base = float(np.mean([distinct_2(x) for x in base_texts]))
    d2_steer = float(np.mean([distinct_2(x) for x in steer_texts]))
    rr_base = float(np.mean([repeat_rate(x) for x in base_texts]))
    rr_steer = float(np.mean([repeat_rate(x) for x in steer_texts]))
    wus_base = float(1.0 - rr_base)
    wus_steer = float(1.0 - rr_steer)

    print(f"[RomanAILabs] Runs: {trials} baseline vs {trials} steered6D")
    print(f"[RomanAILabs] Tokens per run: {n_predict} (EXACT)")
    print(f"[RomanAILabs] Baseline avg total: {b_avg:.3f}s | {b_tps:.2f} tok/s")
    print(f"[RomanAILabs] Steered6D avg total: {s_avg:.3f}s | {s_tps:.2f} tok/s")
    print(f"[RomanAILabs] TRUE overhead avg:  {overhead_avg:+.2f}%  (positive = slower)")
    print(f"[RomanAILabs] TPS change avg:     {tps_delta_avg:+.2f}%  (positive = faster)")
    print(f"[RomanAILabs] distinct_1     base={d1_base:.4f} steer6D={d1_steer:.4f} delta={(d1_steer-d1_base):+.4f} (better=higher)")
    print(f"[RomanAILabs] distinct_2     base={d2_base:.4f} steer6D={d2_steer:.4f} delta={(d2_steer-d2_base):+.4f} (better=higher)")
    print(f"[RomanAILabs] repeat_rate    base={rr_base:.4f} steer6D={rr_steer:.4f} delta={(rr_steer-rr_base):+.4f} (better=lower)")
    print(f"[RomanAILabs] word_uniq_share base={wus_base:.4f} steer6D={wus_steer:.4f} delta={(wus_steer-wus_base):+.4f} (better=higher)")
    print(f"[RomanAILabs] avg_s (sanity)  base=0.0000 steer6D={float(np.mean(steer_avg_s)):.4f}")
    print(f"[RomanAILabs] avg_fces        steer6D={float(np.mean(steer_avg_fces)):.4f}")
    print("\n[RomanAILabs] Done\n")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lighthouse_whitehole_fces_fpmf_6d",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = p.add_subparsers(dest="cmd")

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", type=str, default=None, help="Path to a .gguf model")
        sp.add_argument("--model-dir", type=str, default=None, help="Directory to search for a .gguf")
        sp.add_argument("--pick-regex", type=str, default=None, help="Regex to pick a model from --model-dir")
        sp.add_argument("--prompt", type=str, required=True, help="Prompt text")
        sp.add_argument("--n-ctx", dest="n_ctx", type=int, default=4096)
        sp.add_argument("--n-predict", dest="n_predict", type=int, default=96)
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=0)
        sp.add_argument("--seed", type=int, default=0)

        sp.add_argument("--temp", type=float, default=0.75)
        sp.add_argument("--top-k", type=int, default=40)
        sp.add_argument("--top-p", type=float, default=0.95)
        sp.add_argument("--min-p", type=float, default=0.05)
        sp.add_argument("--typical-p", type=float, default=1.0)
        sp.add_argument("--repeat-penalty", type=float, default=1.0)
        sp.add_argument("--frequency-penalty", type=float, default=0.0)
        sp.add_argument("--presence-penalty", type=float, default=0.0)
        sp.add_argument("--tfs-z", type=float, default=1.0)
        sp.add_argument("--mirostat-mode", type=int, default=0)
        sp.add_argument("--mirostat-tau", type=float, default=5.0)
        sp.add_argument("--mirostat-eta", type=float, default=0.1)
        sp.add_argument("--no-penalize-nl", action="store_true", help="Disable newline penalization")

    sp_run = sub.add_parser("run", help="Run once")
    add_common(sp_run)
    sp_run.add_argument("--no-steer", action="store_true", help="Disable steering")
    sp_run.add_argument("--exact-tokens", action="store_true", help="Force EXACT tokens by banning EOS (recommended for testing)")

    sp_bench = sub.add_parser("bench", help="Benchmark baseline vs steered (exact tokens)")
    add_common(sp_bench)
    sp_bench.add_argument("--trials", type=int, default=5)
    sp_bench.add_argument("--show-excerpts", action="store_true")

    return p

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # UX fix: if you forget the subcommand, assume "run"
    if not argv or (argv[0] not in ("run", "bench", "-h", "--help")):
        argv = ["run"] + argv

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_path = resolve_model_path(args.model, args.model_dir, args.pick_regex)

    if args.cmd == "run":
        print_header("Run (Lighthouse FCES + WhiteHole + RAIL-FPMF 6D)")
        print(f"[RomanAILabs] Model: {model_path}")
        print(f"[RomanAILabs] Prompt: {args.prompt}\n")

        res = run_once(
            model_path=model_path,
            prompt=args.prompt,
            n_ctx=args.n_ctx,
            n_predict=args.n_predict,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
            seed=args.seed,
            steer=not args.no_steer,
            exact_tokens=bool(args.exact_tokens),
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            typical_p=args.typical_p,
            temperature=args.temp,
            repeat_penalty=args.repeat_penalty,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            tfs_z=args.tfs_z,
            mirostat_mode=args.mirostat_mode,
            mirostat_tau=args.mirostat_tau,
            mirostat_eta=args.mirostat_eta,
            penalize_nl=not args.no_penalize_nl,
            fcfg=WhiteHoleFormulaConfig(),
            scfg=SteeringConfig6D(),
            fces_cfg=FCESConfig(),
            fpmf_cfg=FPMFConfig(),
        )

        print(f"[RomanAILabs] Done: {res.elapsed_s:.3f}s | {res.tok_s:.2f} tok/s | tok={res.n_tokens}")
        if not args.no_steer:
            print(f"[RomanAILabs] Steering sanity: hook_calls={res.hook_calls} avg_s={res.avg_s:.4f} avg_fces={res.avg_fces:.4f}")
        print("\n---\n" + res.text + "\n---\n")
        return 0

    if args.cmd == "bench":
        bench(args, model_path=model_path)
        return 0

    parser.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

