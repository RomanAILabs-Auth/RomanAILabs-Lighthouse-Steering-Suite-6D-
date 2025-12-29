# RomanAILabs WhiteHole Steering Module (FCES + WhiteHole + RAIL-FPMF 6D)
Copyright Daniel Harding - RomanAILabs  
Credits: OpenAI GPT-5.2 Thinking
#romanailabs@gmail.com

A **drop-in runtime steering module** for *any* LLM runner that can expose a **per-token logits hook**.  
It computes an online control signal from sampling dynamics and applies **bounded, clipped logits shaping** (plus optional EOS handling) to reduce degeneration (loops / stuck low-entropy / over-dominant top token) while staying measurable and runner-agnostic.

---

## What this module is
- A **sampling-time controller**: it adjusts logits *at each token step*.
- **No training**. No finetuning. No RLHF. No weight updates.
- Designed to be **safe-by-construction**: bounded outputs, clips, stable numerics.

---

## Quickstart
### Install
```bash
pip install numpy
# Optional if you want the llama-cpp adapter:
pip install llama-cpp-python
```

### Run the built-in tester (self-test)
This file ships with a tiny internal self-test. It verifies:
- output shape and finiteness
- EOS ban behavior in exact-token mode
- that stats accumulate sanely

Run it:

```bash
python3 "RomanAILabs_Whitehole_Steering_Module.py"
```

If you renamed it to the recommended module name:

```bash
python3 romanailabs_steering_6d.py
```

Expected output resembles:
- `[RomanAILabs] self-test OK: {'hook_calls': ..., 'avg_abs_s': ..., 'avg_y': ..., 'avg_fces': ...}`

---

## Integration (runner-agnostic)
Your runner must provide, each token step:
- `input_ids`: token history (list[int] or np.ndarray)
- `logits`: numpy array of logits shape `[vocab]`

You create a processor once, then call it every step:

- `processor(input_ids, logits) -> new_logits`

If your runner has:
- an EOS token id, pass it (`eos_token_id`)
- a detokenizer, pass it (`detokenize_fn`) to enable punctuation-aware EOS nudging

This module also provides a convenience adapter for `llama-cpp-python` (if installed).

---

## The math (the “why” in equations)

### 1) Sampling signals
At token step \(t\), compute three signals (all normalized to \([0,1]\)):

**Repetition ratio** over a sliding window \(W\):
\[
r_t = 1 - \frac{|\text{unique}(\text{tail}_W)|}{|\text{tail}_W|}
\]

**Top-k normalized entropy** (using top-k logits, softmaxed):
\[
e_t = \frac{-\sum_{i=1}^{k} p_i \log p_i}{\log k}
\quad,\quad
p=\text{softmax}(\ell_{topk})
\]

**Top-token share** (dominance proxy):
\[
u_t = \max_i(p_i)
\]

---

### 2) Lighthouse FCES (Field–Curvature Entanglement Scalar)
Define a “field” vector \(F_t\) and “curvature” vector \(C_t\) in \(\mathbb{R}^3\):

\[
F_t =
\begin{bmatrix}
w_{rep}\,r_t\\
w_{low}\,(1-e_t)\\
w_{top}\,u_t
\end{bmatrix}
\qquad
C_t =
\begin{bmatrix}
w_{dr}\,\Delta r_t\\
w_{de}\,\Delta e_t\\
w_{du}\,\Delta u_t
\end{bmatrix}
\]

with deltas:
\[
\Delta r_t = r_t - r_{t-1}
\quad
\Delta e_t = e_t - e_{t-1}
\quad
\Delta u_t = u_t - u_{t-1}
\]

Entanglement is computed via the cross-product magnitude:
\[
\sin(\theta_t)=\frac{\|F_t \times C_t\|}{\|F_t\|\,\|C_t\|+\epsilon}
\]

And FCES is a magnitude-weighted scalar:
\[
\mathrm{FCES}_t = g\cdot \sin(\theta_t)\cdot \|F_t\|\,\|C_t\|
\]
Then clipped to a safe range.

**Intuition:** if the “field” and its “curvature” are non-aligned (high cross magnitude), sampling dynamics are behaving in a more coupled / unstable way → steer more.

---

### 3) Quality pressure \(y\)
A single scalar “pressure” is formed:
\[
y_t = b + a_r\,r_t + a_e\,(1-e_t) + a_f\,\mathrm{FCES}_t
\]

---

### 4) WhiteHole Steering \(s \in [-1,1]\)
Let \(x_t\) be a progress-like coordinate (typical choice):
\[
x_t = \frac{t}{n_{predict}}
\]
(or a constant if configured).

WhiteHole computes a heavy-tailed bounded steering scalar:
\[
s_t = \tanh\left(\operatorname{sign}(S_t)\,\ln(1+|S_t|)\right)
\]

where:
\[
S_t = \frac{\sum_{i=1}^{N} a_1\, f(i)\,\sigma\!\left(f(i)K(i)T(x_t,i)T(y_t,0)\exp(-iT(x_t,i)T(y_t,0))\right)}{\sum_{i=1}^{N} f(i)}
\]

with:
\[
f(i) = (i+1)^{-\alpha}
\qquad
K(i) = (i+1)^{-\beta}
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}
\]

**Intuition:** bounded output, heavy-tail response, stable nonlinearity.

---

### 5) RAIL-FPMF (Flux–Potential–Memory Field) 6D controller
Maintain a 6D state integrating the signals:
\[
v_t = [r_t,\ (1-e_t),\ u_t,\ \Delta r_t,\ \Delta e_t,\ \Delta u_t]
\]

A small dynamical system updates internal state:
\[
\text{state}_t = d\cdot \text{state}_{t-1} + dt\cdot \text{inc}(v_t)
\]
(where implementation remaps the entropy delta term for stability.)

It emits:
- a steering strength \(\lambda_t\) (clipped)
- optional EOS bias when “done” is detected (entropy low + punctuation + minimum tokens)

---

### 6) Logits shaping (the control action)
Given logits \(\ell\) and probabilities \(p=\text{softmax}(\ell)\), apply a bounded update:
\[
\ell' = \ell - \operatorname{clip}(\lambda_t \cdot s_t \cdot p,\ -c,\ +c)
\]

**Speed optimization:** by default, shaping is applied only to the **top-N logits** (where probability mass lives), avoiding full-vocab work.

**Exact-token mode:** EOS can be hard-suppressed so output length is exactly \(n_{predict}\). This makes timing comparisons honest.

---

## Why this is “world class” (practical engineering points)
- **Bounded control signal**: \(s\in[-1,1]\), plus log/tanh compression.
- **Clipped actuation**: logits deltas are clipped so nothing can spike.
- **Measurable**: built-in stats (hook calls, average |s|, average FCES).
- **Runner-agnostic**: only needs tokens + logits.
- **Benchmark-friendly**: exact-token mode avoids “EOS ended early” lies.

---

## Common workflows

### A) I want to verify it works (self-test)
```bash
python3 "RomanAILabs Whitehole Steering Module.py"
```

### B) I want to use it with llama.cpp (llama-cpp-python)
Install:
```bash
pip install llama-cpp-python numpy
```
Then create a llama-cpp compatible processor using the module’s adapter (see module API: `make_llama_cpp_processor`).

### C) I want honest performance numbers
Use exact-token mode in your runner (or the reference runner script that bans EOS) so baseline and steered runs generate the **same number of tokens**.

---

## Notes / guardrails
- This improves *sampling dynamics*, not truth. It can reduce repetition and improve flow, but it does not guarantee factual accuracy.
- If you want lower overhead:
  - reduce the WhiteHole loop count \(N\)
  - reduce shaped top-N logits
  - reduce entropy top-k
- Keep the clipping enabled unless you’re deliberately experimenting.

---

## RomanAILabs
This module is designed to be the “steering core” you can inject into any LLM runner.  
If you build on it: keep it bounded, clipped, and benchmarked with exact-token comparisons.
