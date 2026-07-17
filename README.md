# GNoME Structural GNN — Reproducing a Nature Paper on a Budget

This is my attempt at reproducing the structural GNN from Merchant et al.,
*"Scaling deep learning for materials discovery,"* Nature 624 (2023) — the
GNoME paper — at a scale that actually fits on consumer hardware, then
evaluating it against the [Matbench Discovery](https://matbench-discovery.materialsproject.org/)
benchmark.

**Why I'm doing this:** to actually learn how a materials-discovery GNN gets
built, trained, evaluated, and diagnosed — not just read about it. This
isn't a polished, peer-reviewed submission. It's a learning project, and I've
tried to be upfront below about every place I deviated from the paper and
why.

---

## What the model actually does

It predicts the **formation energy per atom** of a crystal structure — but
only structures that have already been *relaxed* (settled into their
low-energy geometry). That's one of two model types the GNoME paper
describes; the other is a NequIP-style potential that also predicts forces
along relaxation trajectories, which is a separate beast I haven't built
(it shows up in the Extensions section below).

---

## Stages of the project (what I've actually done so far)

### 1. Baseline: a plain 200-epoch run

Started with a straightforward single-model training run — 200 epochs, no
bells and whistles — just to get a working pipeline end to end and a first
number to compare everything else against.

### 2. Hyperparameter search on a stratified Kaggle subset

Rather than guess at architecture choices, I ran an anchored coordinate
search (one axis at a time, always carrying the current best forward) on
Kaggle's free T4×2 GPUs, using a stratified subset of the training data
small enough to search quickly.

The satisfying part: the search **independently landed on the paper's own
configuration on every axis** — SiLU activation beat GELU and Mish, 3 GNN
layers won, and a learning rate of 5.5e-4 beat the other candidates I
tried. That gave me real confidence the reproduction was pointed the right
direction before committing to expensive full runs.

### 3. Scaling down on purpose: 6 seeds / 500 epochs instead of 10 / 1,000

The paper trains a 10-member ensemble for 1,000 epochs each. I don't have
that kind of compute lying around — I have a friend's RTX 4070 Ti. So I
made a deliberate, budget-driven call: **6 seeds, 500 epochs**. This is the
single biggest scope reduction in this project, and I want it stated
plainly rather than buried — it's a real deviation, not a paper-faithful
detail.

### 4. Adding EMA weight averaging (my own addition, not in the paper)

Somewhere along the way I added exponential moving average (EMA, decay
0.999) over the model weights. Worth being clear: **this is not in the
paper's training recipe at all.** I tested it against no-EMA and against a
softer EMA-0.99, and 0.999 won cleanly across every metric I checked. So
it's staying in — but it's an improvement I bolted on, not something I
reproduced.

### 5. Test-time augmentation (TTA), and learning the hard way that "mean" isn't always right

The paper uses 20-point volume-scaled TTA at inference time — I implemented
that. But I initially averaged (`mean`) the 20 predictions per structure,
which is the intuitive thing to do. Turns out the paper's real fix for the
domain shift (see below) is to take the **minimum** across those 20 scaled
predictions, not the mean — this flips the systematic bias and meaningfully
improves the F1-style classification metric. Lesson learned: read the
methods section twice.

### 6. Training the 6-seed ensemble, for real

All 6 seeds finished cleanly on the 4070 Ti (one at a time, no fancy
multi-GPU tricks — just serial training). Final per-seed test MAEs came out
tight: 23.32, 24.02, 24.09, 24.33, 23.78, 24.41 meV/atom — mean around 24,
spread of about 1.1 meV. No seed diverged or needed a redo.

### 7. Evaluating on WBM, and catching a real aggregation bug

Ran the full ensemble against WBM's 256,939 initial (unrelaxed) structures.
First pass, I combined the 6 models' predictions with a **mean**. Then I
went back to the paper's methods text and found it explicitly uses a
**median** across ensemble members — specifically because a GNN can be
wildly wrong on out-of-distribution inputs, and one bad model can poison a
mean in a way it can't poison a median. Caught it, fixed it, re-ran. Final
numbers below reflect the corrected median aggregation.

### 8. Diagnosing why the WBM score isn't great

Went digging into *why* the F1-style stability classification score was
low, and found three separate, distinct causes (not just "the model is
bad"):

- **Domain shift** — trained on relaxed structures, tested on unrelaxed
  ones. TTA (min-aggregated) helps a lot but doesn't fully close it.
- A meaningful chunk of false positives are structures that are clearly,
  confidently unstable (way above the hull) — the model is overconfident
  in the wrong direction on these.
- False-positive rate climbs as element count goes up, which tracks with
  the training data (2018 Materials Project) being thin on
  quaternary/quinary compositions.

I also confirmed post-hoc bias correction (just shifting predictions by a
constant) doesn't meaningfully fix the classification score — so this is a
"needs retraining" problem, not a "needs recalibration" problem.

---

## Where things stand — results

### In-distribution (held-out Materials Project test set)

Per-seed MAE, meV/atom: 23.32 / 24.02 / 24.09 / 24.33 / 23.78 / 24.41

### Out-of-distribution (WBM, 256,939 structures)

| Model | MAE (meV/atom) | RMSE (meV/atom) | Bias (meV/atom) | F1 (raw) | F1 (bias-corrected) |
|---|---|---|---|---|---|
| Single-seed baseline (200ep, mean-TTA) | 168.76 | 253.94 | +122.48 | 0.365 | — |
| Single-seed, +EMA-0.999 (mean-TTA) | 159.93 | 244.88 | +110.81 | 0.363 | — |
| Single-seed, +EMA-0.999 (min-TTA) | 109.40 | 169.98 | −47.65 | 0.3432 | 0.3488 |
| **6-seed ensemble, median aggregation, min-TTA (current best)** | **104.71** | **165.46** | **−47.54** | **0.3429** | **0.3486** |

For reference (pulled from a live web search, not the original paper, so
treat as a snapshot rather than gospel): typical universal-potential models
on the current Matbench Discovery leaderboard — MACE, CHGNet, SevenNet and
similar — sit around F1 ≈ 0.57–0.83, with the best model around 0.93. My
0.34 sits below that range, which lines up with the domain-shift diagnosis
above: those models are generally trained on relaxation-trajectory data,
and mine, so far, isn't.

---

## Repo layout

```
analysis/               Per-stage analysis scripts (pattern: run(repo_root, show) -> summary dict)
  stage1_baseline/
  stage1_stageA/
  stage1_ensemble/
configs/
notebooks/
  analysis_gnome_struct.ipynb     Main analysis notebook
  bias_correction_diagnostic.ipynb
  wbm_FP_diagnosis.ipynb
results/                Saved plots and summary tables per stage
runs/                   Training run outputs, checkpoints, predictions, metrics
scripts/
  eval_wbm.py                     Single-model WBM evaluation, 20-pt volume TTA
  eval_wbm_ensemble.py            6-seed ensemble WBM evaluation (median aggregation)
  eval_wbm_ensemble_from_preds.py Fast re-aggregation from existing per-seed CSVs
  f1_wbm.py                       Stability classification F1 / precision / recall
  relax_wbm_with_mlip.py          MLIP pre-relaxation of WBM structures (see Extensions)
src/gnome/
  train_ensemble_seed.py
  graphs.py, model.py
run_ensemble.ps1        Orchestrates 6-seed ensemble training
```

---

## Compute I used

- A friend's RTX 4070 Ti (Windows/PowerShell) — ensemble training, one seed
  at a time
- Kaggle T4×2 — the architecture search
- My own laptop — analysis and notebooks
- Quantum Espresso (PBE) — planned for future DFT work, since I don't have
  a VASP license

---

## Credits

- Big thanks to [Rajib](https://github.com/RajibTheKing) — a Doctoral
Researcher in Real-Time / Embedded Systems in CAU — who let me run the ensemble
training on his RTX 4070 Ti and helped debug some gnarly bugs along the
way (including catching issues in the ensemble training/eval scripts).
This project would have struggled with budget and slowed down significantly without that GPU and that second
pair of eyes.

---

## Extensions

### Currently working on — MLIP pre-relaxation of WBM structures

Instead of retraining the model, I'm testing whether relaxing WBM's
*unrelaxed* structures first — using a pretrained, open universal MLIP —
before scoring them with my ensemble closes some of that domain-shift gap.
Testing two MLIPs side by side for comparison: **CHGNet** and **MACE-MP-0**.

This doesn't touch the trained model at all — it's a pre-processing step
in front of the existing evaluation pipeline
(`scripts/relax_wbm_with_mlip.py`), which relaxes the structures then
automatically runs them through the same ensemble + F1 scoring as above, so
the numbers are directly comparable. Both MLIPs have passed small-sample
smoke tests; the full 256,939-structure run is queued up next, on either
my friend's GPU or Kaggle.

### Planned next

- **Retraining with MPtrj trajectory data** — teaching the model what
  intermediate, non-relaxed structures actually look like, instead of only
  ever showing it finished ones. This is the more fundamental fix for the
  domain-shift problem, and probably the biggest lever left for closing the
  gap to leaderboard-competitive F1 scores.
- **A real active-learning loop** — generate candidate structures, filter
  them with the ensemble, verify the survivors with actual DFT (Quantum
  Espresso), fold the results back into training, repeat. Mostly a
  from-scratch learning exercise in the acquisition/retrain cycle itself,
  likely on a small, narrow chemical system.
- **Matbench Discovery leaderboard submission** — the eventual goal, most
  likely using whichever version of the model (current ensemble, MLIP-
  corrected, or MPtrj-retrained) ends up scoring best.
