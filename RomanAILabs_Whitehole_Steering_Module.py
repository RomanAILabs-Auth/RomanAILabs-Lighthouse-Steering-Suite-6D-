#!/usr/bin/env python3
# Copyright Daniel Harding - RomanAILabs
# Credits: OpenAI GPT-5.2 Thinking
"""
RomanAILabs Steering 6D (FCES + WhiteHole + RAIL-FPMF)
======================================================

A drop-in, runner-agnostic *module* for runtime steering of LLM sampling via logits shaping.

Why this exists
---------------
Most “steering” code is glued to one runner. This module is *injectable*:
- If your runner exposes a per-token logits hook: use it directly.
- If you're on llama-cpp-python: we provide a ready-made LogitsProcessor.

Core components
---------------
1) Lighthouse FCES (Field–Curvature Entanglement Scalar)
   - Online scalar from sampling signals (repetition, entropy, top-share) + their curvature (deltas).
   - Higher FCES ≈ more “non-separable” dynamics → steer harder.

2) WhiteHole Steering
   - Heavy-tailed bounded steering scalar s ∈ [-1, 1] from (x, y).

3) RAIL-FPMF (Flux–Potential–Memory Field) 6D controller
   - Small dynamical controller that produces:
       - steering strength λ (logits shaping magnitude)
       - optional EOS nudge for brevity when “done” is detected

Design goals
------------
- Runner-agnostic: pure Python + numpy.
- Fast enough: shapes only the top-N logits by default (configurable).
- GitHub-grade: clean API, typed, documented, self-test included.

Install
-------
pip install numpy
# llama-cpp-python is optional unless you use the llama adapter:
pip install llama-cpp-python

Quick usage (generic runner)
----------------------------
You need two pieces from your runner each step:
- input_ids: token history (list[int] or np.ndarray)
- logits: current token logits (np.ndarray)

Example pseudocode:
    engine = SteeringEngine6D()
    hook = engine.make_processor(eos_token_id=my_eos_id, exact_mode=False)

    while ...:
        logits = model.forward(...)
        logits = hook(input_ids, logits)
        next_tok = sample(logits, ...)
        input_ids.append(next_tok)

Usage (llama-cpp-python)
------------------------
    from llama_cpp import Llama, LogitsProcessorList
    from romanailabs_steering_6d import SteeringEngine6D

    llm = Llama(model_path="model.gguf", logits_all=True, verbose=False)
    engine = SteeringEngine6D()
    proc = engine.make_llama_cpp_processor(llm, n_predict=128, exact_mode=True)
    lp = LogitsProcessorList([proc])

    # pass lp into llm.sample(logits_processor=lp)

License
-------
Add your preferred license in your repo (MIT/Apache-2.0/etc).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Protocol, runtime_checkable

import numpy as np

ArrayF = np.ndarray


# -----------------------------------------------------------------------------
# Utilities
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


def softmax_topk_probs(logits: ArrayF, k: int) -> Tuple[ArrayF, ArrayF]:
    """
    Return (idx, probs) for the top-k logits (k>=2), stable and float64.
    idx: shape (k,), probs: shape(k,)
    """
    k = int(max(2, min(k, int(logits.shape[-1]))))
    idx = np.argpartition(logits, -k)[-k:]
    top = logits[idx].astype(np.float64, copy=False)
    m = float(np.max(top))
    y = top - m
    np.exp(y, out=y)
    s = float(np.sum(y))
    if s <= 0.0 or not np.isfinite(s):
        probs = np.full((k,), 1.0 / float(k), dtype=np.float64)
    else:
        probs = (y / s).astype(np.float64, copy=False)
    return idx.astype(np.int64, copy=False), probs


def entropy_from_probs(p: ArrayF, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(np.float64, copy=False), eps, 1.0)
    return -float(np.sum(p * np.log(p)))


def entropy_topk_from_logits(logits: ArrayF, k: int = 32) -> float:
    """
    Normalized entropy in [0,1] over top-k.
    """
    idx, p = softmax_topk_probs(logits, k=k)
    _ = idx
    h = entropy_from_probs(p)
    hmax = math.log(float(len(p)))
    return float(h / hmax) if hmax > 0 else 0.0


def top_token_share_from_logits(logits: ArrayF, k: int = 64) -> float:
    """
    Approximate top token probability using top-k softmax (k defaults 64).
    """
    _, p = softmax_topk_probs(logits, k=max(2, int(k)))
    return float(np.max(p)) if p.size else 0.0


def repetition_ratio(tokens: Sequence[int], window: int = 64) -> float:
    """
    0.0 = fully unique in window; 1.0 = fully repeated.
    """
    if not tokens:
        return 0.0
    w = int(max(1, window))
    tail = tokens[-w:]
    if len(tail) <= 1:
        return 0.0
    uniq = len(set(tail))
    return float(1.0 - (uniq / float(len(tail))))


# -----------------------------------------------------------------------------
# 6D Vector
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
# Lighthouse FCES
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FCESConfig:
    # Field weights
    w_rep: float = 1.00
    w_lowent: float = 0.90
    w_topshare: float = 0.60

    # Curvature weights (deltas)
    w_drep: float = 0.80
    w_dent: float = 0.80
    w_dtop: float = 0.40

    # Scaling
    eps: float = 1e-9
    fces_gain: float = 1.00
    fces_clip: float = 2.00


class LighthouseFCES6D:
    """
    Online FCES estimator.

    Field vector F (3D): [rep, (1-ent), top_share]
    Curvature  C (3D): deltas of the above

    Entanglement scalar: sin(theta) * (||F||*||C||)
      sin(theta)=||F x C||/(||F||*||C||+eps) in [0,1]

    Output is clipped to [-fces_clip, +fces_clip] (practically non-negative).
    """

    def __init__(self, cfg: FCESConfig = FCESConfig()):
        self.cfg = cfg
        self._prev_rep = 0.0
        self._prev_ent = 0.0
        self._prev_top = 0.0
        self._init = False

    def reset(self) -> None:
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

        F = np.array(
            [self.cfg.w_rep * rep, self.cfg.w_lowent * lowent, self.cfg.w_topshare * top_share],
            dtype=np.float64,
        )
        C = np.array(
            [self.cfg.w_drep * drep, self.cfg.w_dent * dent, self.cfg.w_dtop * dtop],
            dtype=np.float64,
        )

        Fn = float(np.linalg.norm(F))
        Cn = float(np.linalg.norm(C))
        if Fn > 0.0 and Cn > 0.0:
            cross = float(np.linalg.norm(np.cross(F, C)))
            sin_theta = cross / (Fn * Cn + self.cfg.eps)
            fces = self.cfg.fces_gain * sin_theta * (Fn * Cn)
        else:
            fces = 0.0

        fces = float(clamp(fces, -self.cfg.fces_clip, self.cfg.fces_clip))
        v6 = Vec6(rep, lowent, top_share, drep, dent, dtop)
        return fces, v6


# -----------------------------------------------------------------------------
# WhiteHole Steering
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WhiteHoleFormulaConfig:
    N: int = 512
    a1: float = 1.0
    alpha: float = 1.35
    beta: float = 0.85
    exp_clip: float = 60.0
    inner_clip: float = 60.0


def white_hole_steering(
    x: float,
    y: float,
    cfg: WhiteHoleFormulaConfig,
    T: Callable[[float, int], float],
) -> float:
    """
    s = tanh( sign(S) * ln(1+|S|) )

    S = (Σ a1 f(i) σ(f(i)K(i)T(x,i)T(y,0)exp(-iT(x,i)T(y,0)))) / (Σ f(i))
      f(i)=(i+1)^(-alpha), K(i)=(i+1)^(-beta), σ=sigmoid
    """
    Ty0 = float(T(y, 0))
    numerator = 0.0
    denom = 0.0

    N = int(max(1, cfg.N))
    for i in range(1, N + 1):
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
# RAIL-FPMF 6D Controller
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FPMFConfig:
    enabled: bool = True

    # steering strength (lambda)
    base_strength: float = 2.00
    strength_gain: float = 1.15
    strength_clip: float = 6.00

    # optional temperature model (exposed as a helper)
    base_temp: float = 0.75
    temp_span: float = 0.25
    temp_min: float = 0.50
    temp_max: float = 1.35

    # EOS/brevity heuristics
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
    6D dynamical controller.

    State integrates: (rep, lowent, top, drep, -dent, dtop) with decay and dt.
    """

    def __init__(self, cfg: FPMFConfig = FPMFConfig()):
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
# Steering composition config (ties everything together)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SteeringConfig6D:
    enabled: bool = True

    # quality y composition
    y_bias: float = -0.10
    w_rep: float = 1.20
    w_lowent: float = 0.90
    w_fces: float = 1.15

    # whitehole x input
    x_mode: str = "progress"  # "progress" or "const"
    x_const: float = 0.70

    # repetition / entropy windows
    rep_window: int = 64
    ent_topk: int = 32
    top_share_k: int = 64

    # logits shaping behavior
    logits_shape_clip: float = 10.0
    rep_strength_boost: float = 1.25

    # SPEED: only shape top-N logits (set 0 or <0 to shape full vocab)
    shape_top_n: int = 256

    # exact-token behavior (ban EOS)
    exact_mode_ban_eos: bool = True
    eos_ban_logit: float = -100.0


@dataclass
class SteeringStats:
    hook_calls: int = 0
    sum_abs_s: float = 0.0
    sum_y: float = 0.0
    sum_fces: float = 0.0

    def add(self, s: float, y: float, fces: float) -> None:
        self.hook_calls += 1
        self.sum_abs_s += abs(float(s))
        self.sum_y += float(y)
        self.sum_fces += float(fces)

    def means(self) -> Dict[str, float]:
        d = max(1, int(self.hook_calls))
        return {
            "hook_calls": float(self.hook_calls),
            "avg_abs_s": float(self.sum_abs_s / d),
            "avg_y": float(self.sum_y / d),
            "avg_fces": float(self.sum_fces / d),
        }


# -----------------------------------------------------------------------------
# Runner adapters (optional)
# -----------------------------------------------------------------------------

@runtime_checkable
class Detokenizer(Protocol):
    def detokenize(self, tokens: Sequence[int]) -> bytes: ...


@runtime_checkable
class EOSProvider(Protocol):
    def token_eos(self) -> int: ...


# -----------------------------------------------------------------------------
# The main injectable engine
# -----------------------------------------------------------------------------

@dataclass
class SteeringEngine6D:
    """
    High-level engine that generates an injectable logits hook.

    Defaults are tuned to be:
    - stable
    - measurable (stats)
    - cheap (top-N shaping)
    """
    steering_cfg: SteeringConfig6D = field(default_factory=SteeringConfig6D)
    fces_cfg: FCESConfig = field(default_factory=FCESConfig)
    whitehole_cfg: WhiteHoleFormulaConfig = field(default_factory=WhiteHoleFormulaConfig)
    fpmf_cfg: FPMFConfig = field(default_factory=FPMFConfig)

    def new_components(self) -> Tuple[LighthouseFCES6D, RAILFPMF6D]:
        return LighthouseFCES6D(self.fces_cfg), RAILFPMF6D(self.fpmf_cfg)

    def make_processor(
        self,
        *,
        n_predict: int,
        eos_token_id: int = -1,
        detokenize_fn: Optional[Callable[[Sequence[int]], str]] = None,
        steer_enabled: Optional[bool] = None,
        exact_mode: bool = False,
    ) -> Tuple[Callable[[Union[List[int], ArrayF], ArrayF], ArrayF], SteeringStats]:
        """
        Create a runner-agnostic logits processor:
            processor(input_ids, logits) -> new_logits

        Parameters
        ----------
        n_predict:
            Used only if x_mode == "progress".
        eos_token_id:
            EOS token id if you want EOS ban/nudge. Use -1 to disable EOS handling.
        detokenize_fn:
            Optional for EOS nudge heuristic (punctuation + entropy + min tokens).
            If None, EOS nudging is still possible but will use a minimal fallback text buffer.
        steer_enabled:
            Override steering_cfg.enabled per-processor.
        exact_mode:
            If True, bans EOS to force EXACT tokens (useful for benchmark/testing).
        """
        scfg = self.steering_cfg
        enabled = scfg.enabled if steer_enabled is None else bool(steer_enabled)

        fces, fpmf = self.new_components()
        stats = SteeringStats()

        # tiny local buffer for EOS nudge text tail
        detok_buf: List[int] = []

        def T_func(val: float, i: int) -> float:
            return float(val) * float(i + 1)

        def processor(input_ids: Union[List[int], ArrayF], logits: ArrayF) -> ArrayF:
            # Defensive conversions
            if isinstance(input_ids, np.ndarray):
                history = input_ids.tolist()
            else:
                history = list(input_ids)

            scores = logits.astype(np.float64, copy=False)
            out = scores.astype(np.float64, copy=True)

            # EOS ban for exact mode even if steering off (so the benchmark is truly exact)
            if (not enabled) or (not scfg.enabled):
                if exact_mode and scfg.exact_mode_ban_eos and eos_token_id >= 0:
                    out[eos_token_id] = float(out[eos_token_id] + scfg.eos_ban_logit)
                return out.astype(np.float32, copy=False)

            rep = repetition_ratio(history, window=scfg.rep_window)
            ent = entropy_topk_from_logits(scores, k=scfg.ent_topk)
            top_share = top_token_share_from_logits(scores, k=scfg.top_share_k)

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

            s = float(white_hole_steering(x=x, y=y, cfg=self.whitehole_cfg, T=T_func))
            s = float(clamp(s, -1.0, 1.0))

            lam = fpmf.strength(s_wh=s)
            rep_boost = 1.0 + (rep * max(0.0, scfg.rep_strength_boost - 1.0))
            lam *= float(rep_boost)

            # Shape only top-N logits for speed (default)
            shape_n = int(scfg.shape_top_n)
            if shape_n and shape_n > 0 and shape_n < int(scores.shape[-1]):
                idx = np.argpartition(scores, -shape_n)[-shape_n:]
                top = scores[idx].astype(np.float64, copy=False)

                m = float(np.max(top))
                z = top - m
                np.exp(z, out=z)
                denom = float(np.sum(z))
                if denom <= 0.0 or not np.isfinite(denom):
                    probs = np.full((len(idx),), 1.0 / float(len(idx)), dtype=np.float64)
                else:
                    probs = (z / denom).astype(np.float64, copy=False)

                delta = lam * s * probs
                delta = np.clip(delta, -scfg.logits_shape_clip, scfg.logits_shape_clip)

                # "steer away" from high-prob mass when s positive: subtract delta
                out[idx] = out[idx] - delta
            else:
                # Full-vocab shaping (slower)
                m = float(np.max(scores))
                z = scores - m
                np.exp(z, out=z)
                denom = float(np.sum(z))
                if denom <= 0.0 or not np.isfinite(denom):
                    probs = np.full_like(scores, 1.0 / float(scores.shape[-1]), dtype=np.float64)
                else:
                    probs = (z / denom).astype(np.float64, copy=False)

                delta = lam * s * probs
                delta = np.clip(delta, -scfg.logits_shape_clip, scfg.logits_shape_clip)
                out = out - delta

            # EOS handling
            if eos_token_id >= 0:
                if exact_mode and scfg.exact_mode_ban_eos:
                    out[eos_token_id] = float(out[eos_token_id] + scfg.eos_ban_logit)
                else:
                    # optional EOS nudge
                    if detokenize_fn is not None:
                        # Keep a tail (last ~128 tokens)
                        if history:
                            detok_buf.append(int(history[-1]))
                        tail = detok_buf[-128:]
                        try:
                            text_tail = detokenize_fn(tail)
                        except Exception:
                            text_tail = ""
                    else:
                        text_tail = ""

                    eb = fpmf.eos_bias(token_i=len(history), ent=ent, text_so_far=text_tail)
                    if eb != 0.0:
                        out[eos_token_id] = float(out[eos_token_id] + clamp(eb, -10.0, 10.0))

            stats.add(s=s, y=y, fces=fces_val)
            return out.astype(np.float32, copy=False)

        return processor, stats

    # ----------------------------
    # Llama-cpp-python adapter
    # ----------------------------

    def make_llama_cpp_processor(
        self,
        llm: object,
        *,
        n_predict: int,
        steer_enabled: Optional[bool] = None,
        exact_mode: bool = False,
    ) -> Callable[[ArrayF, ArrayF], ArrayF]:
        """
        Create a llama-cpp-python compatible logits processor.

        Signature matches llama-cpp-python:
            processor(input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray

        Requirements:
        - llm.token_eos() should exist to get EOS id (optional but recommended)
        - llm.detokenize(tokens)->bytes exists for better EOS nudge behavior
        """
        eos_id = -1
        if isinstance(llm, EOSProvider):
            try:
                eos_id = int(llm.token_eos())
            except Exception:
                eos_id = -1

        detok_fn: Optional[Callable[[Sequence[int]], str]] = None
        if isinstance(llm, Detokenizer):
            def _detok(tokens: Sequence[int]) -> str:
                b = llm.detokenize(list(tokens))
                return b.decode("utf-8", errors="ignore")
            detok_fn = _detok

        processor, _stats = self.make_processor(
            n_predict=int(n_predict),
            eos_token_id=int(eos_id),
            detokenize_fn=detok_fn,
            steer_enabled=steer_enabled,
            exact_mode=bool(exact_mode),
        )

        def llama_proc(input_ids: ArrayF, scores: ArrayF) -> ArrayF:
            return processor(input_ids, scores)

        return llama_proc


__all__ = [
    "Vec6",
    "FCESConfig",
    "LighthouseFCES6D",
    "WhiteHoleFormulaConfig",
    "white_hole_steering",
    "FPMFConfig",
    "RAILFPMF6D",
    "SteeringConfig6D",
    "SteeringStats",
    "SteeringEngine6D",
]


# -----------------------------------------------------------------------------
# Self-test (no external deps beyond numpy)
# -----------------------------------------------------------------------------

def _self_test() -> None:
    # Fake logits distribution; verify:
    # - output shape matches
    # - EOS ban works in exact mode
    # - stats update
    vocab = 32000
    eos = 2
    logits = np.zeros((vocab,), dtype=np.float32)
    logits[123] = 5.0
    logits[456] = 4.0
    logits[eos] = 3.5

    engine = SteeringEngine6D()
    proc, stats = engine.make_processor(n_predict=64, eos_token_id=eos, exact_mode=True)

    ids: List[int] = [1, 10, 20, 30]
    out = proc(ids, logits)

    assert out.shape == logits.shape
    assert float(out[eos]) < float(logits[eos]) - 50.0, "EOS should be heavily banned in exact mode"
    assert stats.hook_calls >= 1

    means = stats.means()
    assert "avg_abs_s" in means and "avg_fces" in means

    # Ensure top-N shaping doesn't explode
    assert np.isfinite(out).all()

    print("[RomanAILabs] self-test OK:", means)


if __name__ == "__main__":
    _self_test()

