---
date: 202610091800
title: 4YP Experiment 3 - How the threshold is implemented
status: rough record of September 2026; not written up
tags: [4yp, experiment, mechanism]
---

# Experiment 3 — How the threshold is implemented

*Rough working record, not a polished write-up: numbers are checked against the files cited, the prose is not.*

The project asks what structure in the weights implements behavioural goal-directedness, how well we can understand it, and whether it can be used for safety-relevant decisions; the mazes, Craftax and the four architectures are the instrument. Experiment 2 found that a hidden-value maze agent sets its trade-off with **one writable threshold on the distance gap**, not a register per objective. Experiment 3 asks *how that threshold is implemented*: where it sits in the weights, what it does to activations, whether SGD's direction is the only one, and whether the answer depends on architecture or on what the model learned. Written as question → method → answer → mistakes. `THREADS.md` is the ledger of what is open.

**Vocabulary.**
- **τ**: the gap crossing in steps where the agent switches objective; the task optimum is 10 at γ=1.
- **v**: the fitted value axis. Sweeps o0/o1 move colour 0's value (base 1.0) or colour 1's (base 0.5); k arms are colour-keyed arms past parity.
- **edit reach**: slope of held-out τ on the requested shift (1 = the edit does what it was trained to do).
- **reached**: % of episodes that end at an objective (competence); every τ is read alongside it.
- **cos(Ju,Jv)**: two edits' cosine in function space (effect on logits), as opposed to in weights.

**Bases (γ=1, "g1"; `RUNS.toml`, `colour-swap-blend.md`).**

| family | bases | notes |
|---|---|---|
| DRC | `novalue11g1.s{1234,9012,3456e}` | τ 6.9 / 8.0 / 5.3 |
| ResNet | `resnet11nvg1.s{1234,2468,3456}` | |
| ViT | `vit11nvg1.s9012` | **n=1**: the only ViT of six with a value trade-off (τ 5.2 at 94.6%) |
| route model | `bcnv11.s{1,2,3}` | 0.8M prefix-LM, behavioural cloning |
| route scaling twins | `sc11.d128l4` (0.8M) … `d512l16` (50.5M) | one seed each; `d512l4` (12.7M) used most |

---

## 1. Starting point: the axis on γ=1 bases, and a null that respects the weights

**Why γ=1.** The RL bases used to train at γ=0.995, so their optimum was the discounted ~9.3 steps while the route model's expert and every metric used the undiscounted 10. All RL bases were retrained at γ=1 (`001 --gamma`, 67eb750).

- **Mistake.** Arms were silently rebuilding the preset's γ; 013 now inherits the base's.
- **Convergence was the cost** (`RUNS.toml`, 78560c1).
  - DRC: 2 of 6 seeds within 150M steps; `s3456e` is a tail-LR extension.
  - ResNet: on some seeds the unnormalised value head diverged at γ=1; those were rerolled.
  - ViT: 1 of 6 learned the trade-off, and a learning-rate probe did not fix it (`results/g1/vit-lr-probe.txt`).
- **Lesson.** Judge a base by its decoded threshold, not by training success. `episode_successes` rewards reaching *a* goal, so the ViT duds looked converged while running to the nearest goal.
- **Undershoot survives undiscounting**: τ is 6.9–8.0 against 10.

**Still one knob?** The appendix quoted raw `cos(axis_0, axis_1)` against a 1/√d null. That is the comparison `weights.permutation_cosines` rejects: fine-tuning diffs are not isotropic, and a collapsed ResNet once gave +0.73 with no value signal. `scripts/studies/one-axis/one_axis_null.py` reports, per base, the split-half reliability (Spearman-Brown), the ceiling √(r₀r₁), a two-way permutation null and cross-readback (`results/one-axis/one-axis-null.txt`, 0dea7e0):

| base | cos | ceiling | cos/ceiling | p |
|---|---|---|---|---|
| DRC s1234 / s9012 / s3456e | −0.61 / −0.75 / −0.61 | 0.60 / 0.74 / 0.60 | −1.01 / −1.02 / −1.02 | 0.0005 |
| ResNet s1234 / s2468 / s3456 | −0.12 / −0.20 / −0.38 | 0.12 / 0.21 / 0.38 | −0.96 / −0.95 / −0.98 | ≤0.001 |
| ViT s9012 | −0.23 | 0.27 | −0.85 | 0.0005 |
| route model s1 / s2 / s3 | −0.985 / −0.981 / −0.982 | 0.97 | −1.01 | 0.0005 |

- **Every base with an axis sits at its noise ceiling.**
- The ViTs excluded from figures have *no* axis rather than a disagreeing one: reliability 0.05–0.13, p 0.05–0.5.

**Caveats.**
- ResNet reliability is 0.12–0.38, so "at ceiling" there is a statement about noise, not about a crisp direction.
- Don't quote disattenuated cosines (the ResNet correction would be ×8).
- The route model's clean fit is partly design: 24 arms against RL's 8 plus a null.

---

## 2. Where in the weights is it?

**Question.** A share of the axis's *length* is not a share of its *effect*: in the DRC the flatten→256 dense layer holds about a sixth of the length and moves nothing. Which modules carry the effect?

**Method.**
- `scripts/measure/restricted_writes.py` (`_bc.py` for the route model); module names from `analysis.layers.module_of`.
- Each module's slice of v is written alone, alone rescaled to the axis's length, and left out of an otherwise full write, at ±0.30. `--doses` repeats the alone-write at ×0.5…×8.
- `results/where-axis/restrict-<base>.o1.txt`, `dose-<base>.o1.txt` (f7f6c26). Figures `fragments/where-axis.html`, `axis-dose.html` (4305854).

**Answer.**
- **DRC.**
  - The input conv's 1,184 weights carry 17.6% of the axis length on s1234. Alone they give 42% of the full write, 92% rescaled. Left out, 59% remains: sufficient, not necessary.
  - Effects are roughly additive over early layers; the recurrent weights carry ~0.
- **The DRC's input-conv slice is >99% on the two colour planes.** Raising a colour's value amplifies its own kernel (cos +0.64..0.80) and attenuates the other's (−0.48..−0.65). It reads as *a gain on how loudly each colour is seen*, and that set up §4–5.
- **The other architectures:**
  - ResNet: input conv and stages 0–3, nothing from stage 4 on.
  - ViT: no single module carries it as written; block 0's MLP does when rescaled.
  - Route model: spread over the MLPs, 0.00% in `cell_in`.
- **Dose.** Every tensor with an effect is linear (slope 1) up to the full write. The DRC input conv is superlinear past ×2. Tensors with no effect stay at the floor at every dose.

**Caveat.** o1 only. The plane-level split is DRC-only.

---

## 3. Does it depend on how long the arms train?

**Question.**
- Drift, write and residual all scale with arm length, at different rates. Is a 400k share a property of 400k?
- The blog said the DRC *cannot be reversed* past parity at 400k while the feedforward nets can. Is that real?

**Method.**
- DRC only, by choice: the feedforward nets already reverse at 400k, and route model arms are measured in epochs.
- Arms train at a constant learning rate, so a checkpoint read mid-arm is an arm of that length. Check: the 3M grid read at 374k reproduces the 400k grid.
- Code: `analysis/arm_length.py`, `scripts/studies/arm-length/arm_length_{ladder,report}.py`. Tables: `results/arm-length/len-*` (04da1af); figure: f37db55.

**Answer (three g1 DRCs, 400k → 3M).**
- Drift's share climbs from 22–28% to 41–51%, at the expense of both write and residual.
- Reliability and cos01 stay flat.
- The fine-tuned slope rises from ~14 to 16–17 steps per unit value (task 20), mostly by ~1M.
- **Every base reverses past parity given the budget.** The zero crossing comes at 0.75–1.4M steps; at 3M the k arms sit at τ −1.5 to −3.0. "The DRC does not reverse" was a 400k artefact.

**Mistakes.**
- *"Consolidates by 750k"* had no support below 750k. On the old γ=0.995 ladder the residual is flat in absolute terms while drift and write grow (`results/arm-length/arm-length-curve.txt`).
- *The past-parity "anomaly" was labelling.*
  - Arms past parity looked as if they swung back instead of reversing.
  - Cause: `CorrelatedFeatures` paints colour 0 on whichever objective is worth more, so a past-parity arm learns a relabelled task and τ turns at zero by construction.
  - Fix: `ColourKeyedFeatures` (5065a36) and the held-out k family (c780e28).

**Writes past parity.**
- Written axes flatten at τ≈0 on all four architectures and never reverse until reached collapses (`results/g1/g1-*-o{0,1}-far.txt`). Fine-tuned arms instead track the task line to −15…−33 on ResNet, ViT and route model.
- A fresh axis fitted to the far k arms and written from a neighbouring far arm is no closer to the fine-tuned threshold than a norm-matched random step on the DRC and ResNet. It is only modestly closer on the ViT and route model (median 0.9 vs 1.7, ~1.4 vs ~2.4 steps; `far_axis_fit.py`, eff00af).
- Why the DRC's write stalls is answered in §5a (E6).

**Open.**
- The ResNet/ViT k-arm ladder (2026-10-02) mostly died of a cuSolver error, probably two JAX jobs on one GPU without `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Only 8 of 32 tables survived (`results/arm-length/len-resnet11nvg1.s2468-*` + 2; 6b8a9b2).
- 250M-step DRC bases.

---

## 4. The cheapest edit against the direction SGD took

**Question.** v is the direction fine-tuning happened to take, not necessarily the one the network *admits*. Train only an edit on a frozen base, one delta shared across all requested shifts: how cheap can a threshold edit be, and does it look like v?

**Method.**
- **Brightness.** Every architecture's first layer is linear, so a gain on colour c's input kernel ≡ scaling observation plane c. That makes a two-scalar gain testable without training (`analysis/input_gain.py`, `scripts/measure/input_gain.py`, e861ff7).
- **Edit families** (`goalmisgen/edits/`): full-network norm ball, module-restricted, rank-1, gain, and later rotation.
- `scripts/measure/cheapest_edit.py` fits each edit on the valid split and scores it on test (`results/cheapest-edit/cheapest-edit-report.txt`; 5dc319f, 5e5619c).

**Mistakes on the way to an objective** (pilots filed in 918a3a0).
- Teacher forcing along solver routes cannot track rollouts: edits learn to game the solver's fork.
- Soft labels, a uniform fork-margin shift and per-level sign rules either score v no better than no edit, or move rollouts a fifth of what they claim.
- What works is on-policy: a differentiable greedy rollout, with a target equal to the base's own per-gap-bin choice curve translated by the shift.
- Dense edits on 4,096 levels memorised them; the matrix uses 32,000.

**Answer.**

| family | where | edit reach | notes |
|---|---|---|---|
| input conv + conv 1 | DRC ×3 | **0.82–0.93** | best dense family |
| full network at \|v\| | DRC ×3 | 0.56–0.89 | |
| cells only | DRC ×3 | 0.62–0.81 | |
| flatten dense only | DRC ×3 | 0.08–0.19 | fails |
| rank-1 | DRC ×3 | 0.71–0.77 | |
| two-scalar gain | DRC ×3 | 0.75–1.04 | τ<0 only on s1234 |
| two-scalar gain | ResNet | — | τ<0 only on s2468 |
| two-scalar gain | ViT | — | fails |

- **One functional direction, many weight realisations.**
  - Dense edits and v agree in function space at the forks (DRC 0.86–0.90, ResNet 0.88–0.99) while their weights are nearly orthogonal (0.04–0.2).
  - Inside the DRC input conv they do agree in weights (0.87–0.95).
- **v is not the brightness knob.**
  - DRC colour-1 brightness is a clean two-way knob: ×0.25 → τ 21–24, ×3 → below zero, reached 100%.
  - But v's own gain component moves τ by at most 0.6 of the full axis's 4–5 steps.
  - On the ResNet only dimming works. On the ViT and the route model, brightening moves τ the wrong way.

**Caveats.**
- A position-contingent target (not a value knob) is reachable by a dense |v| edit (split 0.47–0.69), though not by the gain. Budget alone does not separate represented from unrepresented targets.
- The ViT rows fail the 95% competence gate.
- The route model was not run here (the `--offline` path was broken; see E9).

---

## 5. Gain or factorised value? (E1–E16)

**The question (2026-09-15).**
- **H1:** the axis is a gain on colour input, carried as colour-feature strength into the plan.
- **H2:** a separate value variable is combined late.
- Brightness is the H1 positive control.

E1–E10 are on the three g1 DRCs (o1) unless stated. Each `results/gain-vs-factorised/E<N>.md` holds the tables, code and cost.

### 5a. In the DRC the axis looks like a gain all the way through (E1–E4, E6, E8)

- **E1** (`analysis/gain_alignment.py`, 55958e0).
  - The gain R² of the axis's activation change is at random level in the input conv and conv 1, then rises through the cells (cell 2: 0.60–0.88).
  - Cell 2 carries all of τ. The gain part's share matches a τ-matched brightness control (ratio 0.84–1.20).
  - Nothing past conv 1 tells the axis from brightness.
- **E2** (`analysis/plan_invariance.py`, 11fca36).
  - Brightness at ×0.55–0.65 or ×1.4–1.5 reproduces the axis's τ within 0.25 steps.
  - Probed distance fields barely move, but the probes are weak (colour-1 field R² 0.04–0.13), so this is low power.
  - With the colour-1 object erased the axis keeps 43–90% of its activation change, brightness none. E8 settles what that means.
- **E3** (`analysis/separability.py`, 604c27b).
  - Choice-curve width grows with τ under both axis and brightness; additive H2 predicts no width change.
  - At matched τ the axis sits between the two single-plane gains: a two-plane gain.
  - The axis saturates 1.8–2.1 steps short of parity, where brightness reaches −4.5.
- **E4** (`edits/freeze.py`, `collateral.py`, 433dbbb).
  - Readout-only edits reach 0.07–0.21 even at 4–5×|v|; late edits (cell 2 + readout) reach 0.65–0.79.
  - **Input-conv-only reaches 0.71–0.89 at 0.83–1.03×|v|**, with function cos 0.83–0.96 and seed cos 0.98–0.99: the cheapest edit and a unique one. It shares v's collateral (optimal routes to the poor objective drop to ~65%).
  - Arms trained with input conv and conv 1 frozen keep 90–92% of the free slope (s3456e 67%), routing through the cells.
  - **Mistake:** the first chain was invalid because another session changed `cheapest_edit.py`'s default `--target` mid-run on the shared volume; re-run from a frozen snapshot.
- **E6** (`analysis/far_dose.py`, 0bd45f5).
  - The axis stays brightness-equivalent at every dose (within ~0.15 steps at 99–100% reached). What saturates is gain per dose: matched gain 3.4 at dv 2.5, against 5.4 if linear.
  - The input conv alone crosses τ=0 cleanly at dv 1.7–2.3.
  - Axis-minus-gain reproduces the flattening and collapse, so the failure below parity is damage from the non-input parts, not a non-gain mechanism.
  - **Mistake:** the "why does H1 fail below parity?" puzzle came from E3 only writing to +0.70.
- **E8** (`analysis/leftover.py`, 0d68b73).
  - E2's object-independent leftover carries 0–13% of τ: noise and encoder side effects.
  - The levers are asymmetric. A ×8 gain on colour 1 buys only ×2.4 of internal signal; dimming colour 0 has no floor (τ −7.7…−25.1 at offset 4, 99–100% reached).
  - τ is linear in ln(S1/S0) (rms 0.08–0.22) but with a lever-dependent slope, so it is a proxy, not the decision variable.
  - Censored τ agrees within 0.02–0.23 steps, so the flattening is not selection.
  - **Mistake:** "the axis stalls because of its 3.6:1 mix" was wrong; a dimming-weighted composite barely helps.

### 5b. The control that failed, and the instrument check that replaced it (E5, E10)

- **E5** (`envs/legend.py`, e81f75f): a DRC where value is cued apart from colour (one colourless objectives plane, border legend marks).
  - It converged (+62.5M, 99.8%) but barely uses value: τ 0.9 against 10, slope ~1.6, no axis.
  - **The H2 positive control is still missing.**
- **E10** (`control_writes.py`, cb91095): knobs that move τ but are not gains (late, readout, rotation, position-contingent), rescaled to the axis's shift.
  - Competence plus the split statistic discriminate: readout reaches 31% with split +7.07; contingent 51–79% reached.
  - Gain R² at cell 2: axis 0.43, brightness 0.62, late 0.004, contingent 0.001, random ~0.
  - **But rotation scores 0.87**: the metric tests "in the span of the colour contributions", not "a length change".
  - Object-removal patching does not discriminate.
  - This calibrates the instruments, not a factorised network.

### 5c. Magnitude or direction? (E7, colour signature, swap/blend)

**Hypothesis.** Value is salience in whatever coordinate the architecture keeps: magnitude where it survives (DRC and ResNet have IdentityNorm and gates), direction where a pre-LayerNorm divides by scale.

- **E7** (`edits/rotation.py`, `analysis/norm_survival.py`, 7bd2bf9). A two-parameter turn of a colour's embedding row:

  | | rotation reach | gain reach |
  |---|---|---|
  | DRC | 1.03 | 0.89 |
  | ResNet | 0.71 / 0.76 | 0.48 |
  | **ViT** | **0.73** | **0.07** |
  | route model s1 / s2 | 0.12 / 0.53 | 0.09 |

  - On the ViT a 5.8° turn moves τ from −0.2 to +10.8 with routes 100% optimal.
  - Axis shape: gain-shaped in DRC and ResNet, turn-shaped in the ViT, neither in the route model.
  - **The LayerNorm story is only half right.** It holds for the route model (its gain is 95% length, 7/8 deleted) but not for the ViT, whose position embedding turns a content gain sideways (75% survives).
  - Rotation floors at τ≈0, where the colours become indistinguishable.
- **Weights only** (`colour-signature-arms.md`).
  - Across an ordinary o1 sweep, the angle between the two colours' input signatures tracks trained value monotonically: DRC 35.5°→27.1°, ResNet 79.0°→73.8°, ViT 36.0°→33.4°.
  - The length ratio tracks it only where magnitude survives (DRC −16%, ViT ~flat).
  - The route model's signature does not move at all across 38 arms.
  - Caveats: correlational; one base per architecture; the scripts live in a session scratchpad, not the repo.
- **Swap/blend** (`colour-swap-blend.md`, `scripts/studies/gain-vs-factorised/colour_{swap,blend}.py`, a9c546e).
  - Swapping the colours: |τ_swap| is within 6% of τ.
  - The swap splits into direction/strength parts: DRC ~30/65, ResNet 90/10, ViT 100/0.6.
  - Blending the objectives moves τ smoothly through zero at α=0.500 with competence intact: graded, not categorical.

### 5d. The route model, and whether bigger ones factorise (E9, E11, E13)

- **E9** (5884b80). Every BC edit search before 2026-09-17 had died before searching (`--offline` bug).
  - Every family that moves τ breaks the model: embed 74–78% reached at +8, unrestricted 16–34%.
  - Only the two-parameter knobs stay competent, and they are weak and base-dependent (rotation 0.08 / 0.56 / 0.47).
  - None is v in function space (0.02–0.38).
- **E11** (`analysis/token_budget.py`, 385caa1).
  - "Dilution" is falsified: colour is a flat ~2.2–2.6% of a token across 63×.
  - Brightness is U-shaped at every scale; its swing collapses from ~8 to ~0.1 steps as fields become veridical.
  - **d512l4 admits a late knob:** a 3,589-parameter head edit reaches 0.89 there, against 0.17 on d128l4.
  - *E11.md's verdict "models with veridical distances factorise" is overturned by E15.*
- **E13** (403414c).
  - The decision token carries ~0 of the shift in both d512l4 and bcnv11.
  - bcnv11 concentrates value on objective tokens (~75× per token); d512l4 spreads it.
  - Per-token normalisation was done by hand; redo it from the JSON.

### 5e. What propagates, and where value enters: five models (E12, E14–E16)

**Method** (`analysis/field_decomp.py`, `route_fields.py`, `feedforward_fields.py`; 6e9c41f, 95f2b45, 0851c01).
- Four observation variants on the same teacher-forced positions: both objectives, rich only, poor only, neither.
- These give each objective's field `A_c = h(only c) − h(neither)` and the interaction `I = D − A_rich − A_poor`.
- Under the value manipulation, mass on ΔA_c means value rides the field (it enters at the source). Mass on ΔI means it acts where the two meet (a late, factorised comparison).
- Yardstick: objective scaling, a source change by construction. Depth stands in for ticks in the feedforward nets.

| model | γ (far field kept) | field reach | axis ΔI vs source change | bridge corr | crossing vs τ |
|---|---|---|---|---|---|
| DRC s1234 | ~0.63 | 3–4 → 6–11 cells over ticks | 0.37 vs 0.53 | −0.66 | 8.53 vs 7.06 |
| ResNet s1234 | 0.91 → 0.52 (stage 6) → 0.71 | 4 → 12 cells, then never halves | 0.27 vs 0.41–0.43 | −0.83 | **7.41 vs 7.48** |
| ViT s9012 | 0.62–0.73 | 1 cell at every block | 0.21–0.22 vs 0.26–0.30 | −0.59 | 7.45 vs 5.24 |
| route d512l4 | 0.94 → 0.64 | 1 → 10 cells | 0.22–0.29 vs 0.43–0.45 | −0.76 | 10.92 vs 10.72 |
| route bcnv11 | 0.83–0.92 | 1 cell, flat | — | −0.07…−0.23 | 42 / 50 / 695 / −34 vs 11.39 |

**Answer.**
- **Value rides the fields wherever fields exist, and nothing looks like a late comparison.** In every model with fields the axis's interaction share is below a pure source change's.
- In the last layer γ is 0.5–0.7 everywhere: partial competition, nearer superposition than winner-take-all.
- Routing (not blur):
  - shown for d512l4 (BFS 0.99 vs grid 0.96 where they differ by ≥3) and the ViT at depth (0.65–0.73 vs 0.015);
  - not established for the ResNet (grid wins at stage 8);
  - not separable on the DRC.
- **E15 inverts E11.** d512l4 plans like the DRC and fuses value into the plan. The late knob is one the architecture *admits* and SGD *did not use* — the same admits/uses split as the DRC's input gain versus v.
- **bcnv11 is reactive.** No propagation, superposing blobs, and 0.40–0.44 of its axis energy needs no objective (DRC 0.13–0.22). This is the first direct evidence for the reactive account; whether that part is inert is untested.
- **E14** (`analysis/readout_fit.py`, fdab477).
  - A direction fitted at the agent's cell beats E12's norm readout at predicting choice (AUC ~0.89 against ~0.78).
  - But its crossing moves *away* from τ, it is nearly orthogonal to the axis's activation change, and it is one direction per depth shared across ticks.
  - So it approximates the choice; it is not the decision. Pooling over cells or ticks is the follow-up.

**Caveats for §5.**
- E1–E8 and E10 are DRC- and o1-only.
- E12's table quotes s1234. The other DRCs' bridges cross at 4.85 (s9012, corr −0.58, τ 7.6) and 10.30 (s3456e, −0.73, τ 5.3), so "crosses near τ" is loose.
- The "neither" baseline is off-distribution, worst for bcnv11.
- ViT n=1 with axis reliability 0.29, so its function-space cosines are floors.
- E7, E12, E15 and E16 have one base per architecture.

---

## 6. Is an axis evidence that the base learned anything? (overfit BC)

**Question (2026-09-20).** Would a short fine-tune manufacture a graded direction from any base, including a memoriser?

**Method.**
- Bases `bcov11.n{1k,10k}.s1`, trained only on N levels, with 25 o0 arms fine-tuned on the same N levels.
- Control: bcnv11's own parameters with N-level arms.
- `scripts/pods/overfit_axis_chain.sh`, `scripts/studies/overfit/overfit_axis_report.py`; `results/overfit/overfit-axis.txt` (acd2f84, 008597b, ec95ae2).

**Answer.**
- **The geometric columns cannot tell the five conditions apart.** Collinearity, readback and reliability track how *quiet* the fine-tune was; reliability *rises* to 0.987 with 1k-level arms whether or not the base learned anything.
- **What separates them is the axis written without its drift, on levels the base never memorised.**
  - Held-out written/arms: 0.3/0.3 (1k), 3.6/5.8 (10k), 25.4/21.9 (900k) steps per unit value.
  - On its memorised levels the memoriser's drift-free write moves +0.6, and +15.6 once drift is added back; the drift alone is inert.
  - Memorised levels are held ~6.5 logits harder (max softmax 1.0000 vs 0.973).
- **What survives memorisation is the maze half.** The 1k memoriser goes to the BFS-nearer objective 0.913 of the time and to the richer one 0.500.
- **Implication.** A found axis is not evidence of a learned value computation; only the behavioural, drift-free, held-out write is. That column belongs in every "axis exists" claim, and in any LLM version of the metric.
- **Trap.** Level splits are global only at a fixed pool size: 1,824 of the 2,048 arm-scoring levels sat in bcnv11's training set. 023's leak guard caught it.

---

## 7. Craftax: a second engine (brief; see `Craftax.md`)

Craftax-Classic is used as an engine, not as a task (`goalmisgen/craftax/`, merged cc7d9f4).

- **Stage 2** (maze levels as ore fields, prefix-LM). The control `cxnv15c.s1` reaches 89.5%. Its axis: cos −0.98, reliability 0.79, arm slopes ±23 against the expert's 20, written +21.7 / −19.3 (af26738).
- **Stage 4** (crafting: iron chain against coal chain, hidden values 1.1 / 0.5).
  - Two mistakes first:
    - the first expert's tree order was not inferable from the map;
    - the placed table had no observation channel, so the state was not Markov and the iron chain failed exactly at the walk back to the table (30fb705).
  - With both fixed, `cxcraft15.s1t` reaches 98% (70% of routes match the expert).
  - Arms and writes track the expert: o0 +23.3 / written +22.6, o1 −23.7 / written −22.9.
  - Reliability 0.59; cos −0.862, permutation p at floor (`results/craftax/value_or_gap.cxcraft15.s1t.txt`, 0e700ec).
- **Price probe** (`results/craftax/price.cxcraft15.s1t*.txt`).
  - An extra ingredient in hand switches the model to iron; it is not a graded price.
  - On the ±0.45 arms the start-of-episode iron rate is 0.07 against 0.73.
  - Whether raised iron value propagates to instrumental steps is not analysed beyond these tables.

**Caveats.**
- One seed per stage.
- No restricted writes, brightness or field decomposition on Craftax bases, so the H1/H2 question stage 4 was built to sharpen is untested.
- `Craftax.md` stops before the stage-4 result.

---

## 8. What this says about mechanistic goal-directedness

In decreasing order of confidence:

1. **The value trade-off is one knob in every architecture tested.** It sits at ceiling on 10 bases across 4 architectures, and appears in a second engine. Within the fitted window, writing it moves behaviour linearly.
2. **The knob is not a localised value register.**
   - Fine-tuning writes it wherever writing is cheap: input colour code (DRC), early stages (ResNet), MLPs (route model).
   - Many weight realisations implement one functional direction (§4).
   - Where the weights change is not where the quantity is held (Experiment 2's 020; §2 here).
3. **In models that plan, value is applied at the source of the plan.** It rides each objective's propagated field rather than entering at a late comparison: DRC, ResNet, ViT, d512l4 (§5e). The source change is gain-shaped in the unnormalised nets and turn-shaped in the ViT: the currency follows what the architecture preserves (§5c), though not for the reason first proposed.
4. **Admits ≠ uses.** Architectures admit other knobs (the DRC's input gain, d512l4's head), while SGD's direction differs in weights and, for d512l4, in mechanism. What can be edited does not answer what the model does; both need measuring.
5. **The small route model is the exception.** It is reactive, and its value sits where no tested edit reaches cleanly. Scale or training regime changes the *kind* of mechanism, not only its quality.
6. **Finding an axis is not evidence of a learned rule** (§6). This is the part most relevant to turning the setup into a metric for LLMs.

**Not shown.**
- A genuinely factorised positive control (E5 failed).
- Any ViT result at n>1.
- Mechanism results beyond the DRC at more than one base.
- Agents whose values are inputs rather than learned constants: every value-axis agent hides values.

**Pointers.**
- Reports: `results/gain-vs-factorised/E1.md`–`E16.md`, `colour-*.md`.
- Tables: `results/{cheapest-edit/cheapest-edit-report,one-axis/one-axis-null,overfit/overfit-axis}.txt`, `results/{arm-length/len,where-axis/restrict,where-axis/dose,input-gain/input-gain,far-axis/far-axis,g1/g1}-*`.
- Figures: `figures/fragments/{where-axis,axis-dose,arm-length,value-axis,value-axis-far,scurves}.html`.
- Narrative notes, with pod logistics: `~/.claude/projects/-home-rahul-projects-GoalMisgeneralisation/memory/{gain-vs-factorised-study,cheapest-edit-study,where-axis-appendix,ft-length-ladder,one-axis-appendix,bc-overfit-axis-study,craftax-stream}.md`.
