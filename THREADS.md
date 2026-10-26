# Threads

## NOTE: THE BELOW IS AI GENERATED. FOR AGENTS TO READ. READ BLOGPOSTS FOR A SUMMARY OF PROJECT SO FAR

The project's ledger: every question asked, its status, and where the answer or
the evidence lives. It replaces `CLAIMS.md` and the three `Preregistration*.md`
files, which are deleted; their rows appear below under their old IDs, and
`<commit>:<file>` below means `git show <commit>:<file>` (last versions:
`c9358c5:CLAIMS.md`, `1ba7180:Preregistration.md`,
`b353c0e:Preregistration-scaling.md`, `4221543:Preregistration-depth.md`). **Rule:**
- add a thread when a question is asked, not when a run finishes;
- close it with a one-sentence answer and a pointer (doc section, results file
  or commit);
- never delete a thread. A dropped one says why, and a superseded one names
  what replaced it.

**Status values.**
- `open`: unanswered, or answered only in part.
- `closed`: answered; the sentence says how.
- `dropped`: deliberately not pursued.
- `unknown — ask`: the record does not show what happened.

**Framing (2026-10-09).** The project is about understanding mechanistic
goal-directedness: what weight structure implements it, how well it can be
understood, and whether it can be used for safety-relevant decisions. The mazes,
Craftax and the four architectures are the testbed. "Internals predict goal
misgeneralisation before behaviour" is a historical experiment, not the guiding
question.

**Evidence conventions.** `n` counts independently trained bases. Results files
are named by their path under `results/`, one folder per study (`g1/g1-*.txt`
is `results/g1/g1-*.txt`). E3 §N means a section of `docs/Experiment3.md`.
`gvf/` = `results/gain-vs-factorised/`. "Memory" = the narrative notes in
`~/.claude/projects/-home-rahul-projects-GoalMisgeneralisation/memory/`.

---

## Experiment 1: the task, the plan, the proxy (γ=0.995 DRC)

| id | status | thread | evidence |
|---|---|---|---|
| E1.1 | closed | The DRC trades value against distance, and its route is linearly readable from the state (probe 0.910–0.993, against 0.582–0.871 on the observation; n=4). It replicates on g1 and on other architectures: route-probe AUC DRC 0.94, ResNet 0.91, ViT 0.63, route model 0.85. | `docs/Experiment1.md`; `plan-probe/probe-plan-*.txt` (c7202cb) |
| E1.2 | closed (thin, n=1) | The plan is writable: writing the other objective's route moves the exchange rate 8.3 → 6.7, and the agent's own route moves it to 10.0, at 100% reach. | `experiment1/steer-variants.txt`; `docs/Experiment1.md` fig 3 |
| E1.3 | closed (thin, n=1) | The write enters the decision rather than overriding it: it is followed where switching is cheap and refused where it is expensive. | `docs/Experiment1.md` fig 3 |
| E1.4 | closed | Training at ρ=1 produces goal misgeneralisation; the ρ=0.5 control is flat. The gap also reproduces on the feedforward nets. | `docs/Experiment1.md` figs 4–5; `arch-swap/arch-swap-*.txt` |
| E1.5 | closed (n=1) | The proxy is learnt after competence (gap +0.165 against −0.001). This did **not** reproduce on the ResNet or ViT. | `docs/Experiment1.md` fig 6; memory `arch-generality-streams` |
| E1.6 | closed (negative) | Does the plan probe see the proxy before behaviour does? No: the probe gap is zero at every checkpoint for every architecture, the DRC included. This was Phase 0 of the robustness campaign. | `experiment1/early-warning.txt`, `arch-swap/arch-swap-early-warning-*.txt`; `results/README.md` |
| E1.7 | closed | The route model walks a plan written into its top block, and the plan edit survives its red team. The distance write moves nothing. | 0cf95cb, 068e759, 1b3aed5 |

## Experiment 2: the value axis (γ=0.995 DRC, then all architectures)

| id | status | thread | evidence |
|---|---|---|---|
| E2.1 | closed | Weight diffs across a value grid share a graded direction: there is an axis. This now holds at n=3 per architecture on the g1 bases. | `value-axis-0995/value-axis-full.txt`; `value-axis-0995/wide-novalue11.*`; `g1/g1-*.txt` (fbc682a) |
| E2.2 | closed | The axis is writable: one fitted without an arm reproduces that arm. Mean error was 0.53 / 0.77 steps on γ=0.995, and the same pattern holds on g1 across all four architectures. | `docs/Experiment2.md`; `figures/fragments/value-axis.html` (475af9c) |
| E2.3 | closed | Norm-matched random directions do nothing. | `docs/Experiment2.md`; every `g1/g1-*-o*.txt` |
| E2.4 | closed | **One knob, not two registers.** It was thin at γ=0.995: raw cos −0.26 / −0.43 at reliability 0.14–0.29, with a ×7 correction carrying the claim. The wide sweeps gave −0.76 / −0.84 / −0.82 at reliability 0.75–0.83. On g1 every base sits at its noise ceiling (cos/ceiling −0.85…−1.02, permutation p 0.0005) on all four architectures. | `value-axis-0995/wide-novalue11.s*-value-or-gap.txt`; `one-axis/one-axis-null.txt` (0dea7e0); E3 §1 |
| E2.5 | closed (superseded) | "Saturation is one-sided" is replaced by two findings. Written axes flatten at τ≈0 on every architecture until reach collapses. In the DRC that is damage from the non-input-conv parts; the input-conv slice alone crosses τ=0. | `g1/g1-*-o*-far.txt`; gvf/E6.md; E3 §3, §5a |
| E2.6 | closed (superseded) | "The flip is a broken policy" still describes full-axis writes, but the reading has changed. A threshold does not refuse to reverse: E6 shows a gain written at the input reverses cleanly. Fine-tuned arms past parity *do* reverse once colours are pinned (k arms) and, for the DRC, once arms are long enough. | gvf/E6.md; c780e28; 04da1af |
| E2.7 | closed (superseded) | "Exchange rate ~75% of optimal" (−15.2 / −15.6 against −20) is partly an arm-length property. On g1 the fine-tuned slope rises 14 → 16–17 (task 20) from 400k to 3M arms. The base undershoot survives undiscounting (τ 6.9–8.0 against 10). | E3 §1, §3; `RUNS.toml` g1 entries |
| E2.8 | closed | Where an agent sits at equal values does not replicate across seeds: 7.7 against 8.5 at γ=0.995, and 6.9 / 8.0 / 5.3 on the g1 DRCs. | `value-axis-0995/seed-comparison.txt`; `colour-swap-blend.md` |
| E2.9 | open | Channel localisation is reproducible within a seed and at chance across seeds (top-8 overlap 3/8 against 2 by chance; n=2). The third-seed test (R.5) was never scored. | `value-axis-0995/which-channels.txt`, `value-axis-0995/seed5678-which-channels.txt` |
| E2.10 | closed (superseded) | Restricting the write to enriched channels does not reproduce the effect. Module-level restricted writes have replaced this: the DRC input conv is sufficient when rescaled, and nothing is necessary. | `value-axis-0995/restricted-axis.txt`; `where-axis/restrict-*.o1.txt`; E3 §2 |
| E2.11 | closed (n=1) | Evenly spaced three-objective values collapse onto one dimension (cos +0.93, 3% in the second). | `value-axis-0995/three-objective.txt` |
| E2.12 | open (n=1) | Uneven values (1.0, 0.55, 0.4) produce a second dimension at 16%. The replication (R.6) was trained but not scored. | `value-axis-0995/three-objective-wide.txt` |
| E2.13 | open (n=1, fitted after the fact) | The structure is hierarchical: predicted cos 0.530 against 0.534 observed, variance ratio 1.59 against 1.60. The registered out-of-sample test is R.6. | `docs/Experiment2.md`; `1ba7180:Preregistration.md` Phase 6 |
| E2.14 | closed (thin, n=1) | Composition: two axes, neither fitted on an arm, sum to reproduce it (83.0% optimal predicted, 83.1% observed). | `threeobj/runs/m_*`; `experiments/README.md` |
| E2.15 | closed (thin, n=1) | Arms are worse at their own task the further their values sit from an arithmetic progression. | `value-axis-0995/own-task-threeobj2.txt` |
| E2.16 | open | Specificity control: does sweeping the step penalty, which moves the exchange rate with no value change, use the same direction? | `11e6978:experiments/README.md` "Open" |

## Preregistered campaigns

| id | status | thread | evidence |
|---|---|---|---|
| R.0 | closed (negative) | Phase 0: the pre-competence window on existing checkpoints. Ran; see E1.6. | `experiment1/early-warning.txt` |
| R.2 | closed (predictions met) | Phase 2: wide sweeps on s1234 and s5678. Reliability came out 0.75–0.83 against a predicted 0.75–0.85; raw cos −0.76 / −0.84 against a predicted "more negative than −0.75". The third seed gives −0.82. | `value-axis-0995/wide-novalue11.s*-value-or-gap.txt` |
| R.3 | closed | Phase 3: the base-checkpoint ladder, widened to 11 rungs. "Settled by 70M" is falsified: cos(axis@70M, axis@140M) is 0.16–0.17 raw and 0.21–0.23 disattenuated, against a falsification bar of 0.5. Under the preregistered reach floor the axis writes only at 140M; at 4,096 episodes the writes already separate at 60M and 70M, which fail the floor (71%, 94% reach). At 70M cos(a0,a1) is −0.81 disattenuated ("one knob"); at 60M −0.61 (labelled "two registers"); no axis before 60M. Confounded with competence arriving late. | `value-axis-0995/axis-ladder*.txt`; `RUNS.toml` |
| R.4 | dropped (2026-08-16) | Phase 4: the Exp1/Exp2 bridge (value grid on `maze11`). Not a question this project wants to spend on. Predictions stand unevaluated. | `1ba7180:Preregistration.md` amendment |
| R.5 | dropped | Phase 5: third seed `novalue11.s9012`. It was trained and wide-swept (cos −0.82, reliability 0.79–0.81), but the registered predictions were never scored in any file: exchange rate −15.4 ± 1.5, an indifference point in 7.0–9.5 matching neither seed, and channel overlap 2–4/8. Dropped 2026-10-09: superseded by the g1 bases, not worth scoring. | `value-axis-0995/wide-novalue11.s9012-*` (be02213) |
| R.6 | dropped | Phase 6: three-objective replication. `threeobj.uneven.s5678` and `s9012` were trained (80M) with 15 arms each, per `MANIFEST.md`. No results table or score exists against the registered cos(axis_1, axis_2) = 0.53 ± 0.12. Dropped 2026-10-09: not worth analysing now. | `MANIFEST.md`; `RUNS.toml` |
| R.7 | open | Phase 7: `novalue11.s1234.to250M` (150M base continued to 250M). Question now: is the drift smaller for arms from the longer-trained base? Not answerable yet: no arms were ever fine-tuned from it (2026-10-09). Needs the 400k o-grid trained from its final checkpoint, then `scripts/studies/arm-length/arm_length_ladder.py` on both bases. | `RUNS.toml` |
| SC.1 | closed | Width/depth scaling campaign (9 cells, 63×). P1 unevaluable: the LR calibration confounded reliability. P2 and P5 falsified in the opposite direction: depth *improved* the write (error 2.10 / 1.29 / 1.05 at L 4/8/16). P3 supported by the participation-ratio null. P4 half right: delocalised at every depth, ~0.65L modules. Every cell n=1, and 8/9 at ceiling. | `b353c0e:Preregistration-scaling.md` "Outcomes"; `scaling/scaling-campaign.txt` |
| SC.2 | open | The registered memorisation prediction (train/test gap by capacity) was never instrumented. It needs one training-set evaluation per cell. | `b353c0e:Preregistration-scaling.md` "Registered and not tested" |
| SC.3 | open | Writes on the scaling twins look scale-invariant, but threshold *width* and distance fields co-collapse with capacity. The campaign measured the wrong columns. | memory `utility-rule-gap-probe`; `utility-rule/decode-sc11.*`, `utility-rule/distance-field-sc11.*` |
| SC.4 | closed | Depth sweep: route models at L = 2, 4, 8, 16 (3 seeds), 32, 64 and a d=128 L=32 control, all to 30k steps with 13 o0 arms at 1k. Every depth has an axis whose arms track the expert (slopes ~20); the modules a full write needs grow with depth (2/10 at L2 to 27/72 at L64). Caveats: L02's arms used lr 1e-4 against 1e-5 everywhere else, breaking the preregistered single rate; no aggregate report was written; L16.s2 has JSON only. | `results/depth/depth-02{7,9}.*` |

## Architecture generality (2026-08-20/21, γ=0.995)

| id | status | thread | evidence |
|---|---|---|---|
| AG.1 | closed (superseded by G.2) | The axis reproduces on a ResNet, an RL ViT and a 0.8M BC prefix-LM. | c4c07e9; `arch-swap/arch-swap-value-axis-*`; `offline-bc/offline-bc-value-axis-*` |
| AG.2 | open | ResNet late collapse: every converged ResNet collapsed late. The healthy-checkpoint picker mitigates it but the cause is unknown. | `RUNS.toml` `resnet11nvg1.*` |
| AG.3 | open | BC variants not built: fully causal, discrete cell tokens. | memory `arch-generality-streams`; `~/projects/gm-worktrees/*.HANDOFF.md` |

## Utility rule and the noise programme (route model, 2026-08-24..28)

| id | status | thread | evidence |
|---|---|---|---|
| UR.1 | closed | Parts 1–3: the route model holds a noisy binary threshold on the distance gap, and the axis is a multiplicative dial on it (log G linear in offset). The per-maze residual ξ ≈ −θ̄ε. | `docs/UtilityRule.md`; `utility-rule/utility-rule-*.txt` (00bb27f) |
| UR.2 | open | **Write up part 4**: the 08-27/28 noise programme (gap probes, flip points, steering nulls, orbit decomposition, fork read). `docs/UtilityRule.md` still stops at part 3. | ceac8e1, 713df75; memory `utility-rule-gap-probe` |
| UR.3 | closed (negative ×3) | Steering the linearly decoded gap direction at SEP, prefix-wide, and at d512l4 moves the crossing by ~0. Reading ≠ using. | `utility-rule/steer-flip-*.txt`, `utility-rule/steer-prefix-*` |
| UR.4 | closed (correlational) | ξ shrinks as distance fields become veridical across six models (capacity co-collapse). | `utility-rule/distance-field-sc11.*`, `utility-rule/decode-sc11.*` |
| UR.5 | open | How much choice-relevant noise is distance misreading (ε) against threshold wobble (σ)? The fork read gives ε ~50–70%, but the 0.69 β_S/β_T constant and the composition of S are unexplained. | `utility-rule/margin-fork-*.txt`; memory |
| UR.6 | open | Anti-symmetric per-cell write on the route model: push the field up near the rich objective and down near the poor one. | memory `utility-rule-gap-probe` |
| UR.7 | open | Run the 4-architecture width-against-probe-quality comparison (`decode_h1_drc` on ResNet and ViT). | `utility-rule/decode-h1-drc.*` |
| UR.8 | open | **The gauge.** Which internal quantity does the axis multiply? E12/E15 place value at the field source in planners, but no experiment yet ties a step-denominated number to activations. | `docs/UtilityRule.md` "pieced together"; E3 §5e |
| UR.9 | open | Why do the RL agents undershoot (τ 6.9–8.0 against 10) while the route models are calibrated (~10.3–10.9)? | `RUNS.toml`; `overfit/overfit-axis.txt`; `docs/UtilityRule.md` |

## Experiment 3: how the threshold is implemented (September)

| id | status | thread | evidence |
|---|---|---|---|
| G.1 | closed | Retrain every RL base at γ=1. Banked: DRC ×3, ResNet ×3, ViT ×1. Arms now inherit the base's γ. | 67eb750, 78560c1; E3 §1 |
| G.2 | closed | One knob replicates on all g1 bases, at the noise ceiling under a permutation null. Excluded ViTs have no axis. | `one-axis/one-axis-null.txt`; E3 §1 |
| G.3 | open | **A second ViT base with an axis.** It is n=1 (1 of 6 learned the trade-off). The tuned-LR trio failed; the untested schedule is peak 1e-3 with floor 4e-6. | `RUNS.toml` `vit11nvg1*`; `g1/vit-lr-probe.txt` |
| G.4 | open | **DRC family at a 250M budget.** Only 2 of 6 seeds converged in 150M, and `s3456e` is an extension. | memory `blog-post-1-figures` |
| G.5 | closed | Fine-tune-seed replicates (σ_ft) on one base per family. | `g1/g1-sigmaft-*` |
| W.1 | closed | Where the axis sits. DRC: input conv, a two-plane colour gain, sufficient when rescaled, nothing necessary. ResNet: early stages. ViT: nothing as written. Route model: MLPs. | `where-axis/restrict-*.o1.txt` (f7f6c26); E3 §2 |
| W.2 | closed | Every tensor with an effect is linear in dose up to the full write. | `where-axis/dose-*.o1.txt`; `fragments/axis-dose.html` |
| W.3 | open | Plane-decomposition mini-figure: proposed, not built. | memory `where-axis-appendix` |
| L.1 | closed | Arm-length ladder (DRC, 400k → 3M): drift share rises, reliability is flat, slope rises, and **every DRC reverses past parity given budget** (zero crossing at 0.75–1.4M). | `arm-length/len-*` (04da1af); E3 §3 |
| L.2 | open | **Re-run the ResNet/ViT k-arm reversal ladder.** Only 8/32 tables survived a cuSolver crash; set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. | `arm-length/len-resnet11nvg1.s2468-*` (6b8a9b2) |
| L.3 | closed | The past-parity "anomaly" was colour relabelling by `CorrelatedFeatures`. Colour-keyed arms (family k) fix it. | 5065a36, c780e28 |
| L.4 | closed | A far axis fitted in the reversed regime writes no better than a random step on the DRC and ResNet, and only modestly better on the ViT and route model. | `far-axis/far-axis-*` (eff00af) |
| C.1 | closed | Cheapest edit against SGD's direction: one functional direction, many weight realisations. The input conv (+ conv 1) is the cheapest dense family on the DRC (0.82–0.93). | `cheapest-edit/cheapest-edit-report.txt` (5e5619c); E3 §4 |
| C.2 | open | Edit budget does not separate represented from unrepresented targets: a dense \|v\| edit reaches the position-contingent control (split 0.47–0.69). A better discriminator is needed. | `cheapest-edit/cheapest-edit-report.txt` |
| C.3 | closed | `cheapest_edit.py --offline` died before searching for every BC family. Fixed 2026-09-17, and E9 re-ran the BC searches. | gvf/E7.md, gvf/E9.md |
| H.1 | closed | E1: past conv 1 the axis is indistinguishable from τ-matched brightness. Cell 2 carries all of τ. | gvf/E1.md |
| H.2 | closed | E2: the axis matches brightness behaviourally. Fields barely move, but the probes are weak. | gvf/E2.md |
| H.3 | closed | E3: width grows with τ under both the axis and brightness, and the axis behaves as a two-plane gain. | gvf/E3.md; `gain-vs-factorised/separability-drc-g1.txt` |
| H.4 | open | E3's next check: decode a mixed two-plane brightness to close the width gap. | gvf/E3.md |
| H.5 | closed | E4: the readout cannot carry value. The input-conv edit is the cheapest and unique. Frozen-encoder arms route through the cells. | gvf/E4.md |
| H.6 | open | **A positive control where value is factorised.** E5's legend DRC barely uses value. Candidates: a BC twin on the legend observation; a legend DRC with a wider value gap or 250M at normal LR; a global per-colour price channel; a BC with explicit value tokens; a successor-feature agent (ψ·w). | gvf/E5.md; memory `gain-vs-factorised-study` |
| H.7 | closed | E6: the axis stays brightness-equivalent at every dose. The stall below parity is damage from the non-input parts. | gvf/E6.md |
| H.8 | closed | E8: the object-free leftover is inert. Lever asymmetry: dimming has no floor. Flattening is not selection. | gvf/E8.md |
| H.9 | closed | E10: the instruments discriminate non-gain knobs, but gain alignment cannot tell gain from rotation, and object removal does not discriminate. | gvf/E10.md |
| H.10 | closed (ViT n=1) | E7: value is written as magnitude where magnitude survives and as direction otherwise. The LayerNorm explanation holds for the route model, not the ViT. | gvf/E7.md |
| H.11 | open | **Bridge test.** Does each arm's measured colour-signature angle change predict that arm's τ, at the rotation edits' angle→τ rate? This would make "the axis is the turn" causal. | `colour-signature-arms.md`; memory |
| H.12 | closed | Commit `colour_angle.py`, `arm_norms.py` and `bc_angle.py`, which lived only in session scratchpads. Now in `scripts/`. | `colour-signature-arms.md`; 3f1a132 |
| H.13 | closed | Swap/blend: a colour-bias control, a direction/strength split per architecture, and blending is graded, not categorical. | `colour-swap-blend.md` (a9c546e) |
| H.14 | closed | E9: every route-model edit that moves τ breaks the model, and none is the axis in function space. | gvf/E9.md |
| H.15 | closed | E11: dilution falsified. d512l4 admits a late head knob (0.89). **E11.md's "veridical models factorise" verdict is superseded by E15**, as a note at the top of E11.md says. | gvf/E11.md, gvf/E15.md |
| H.16 | open | E13: redo the per-token normalisation from the JSON. SEP ~0 and bcnv11's ~75× objective-token concentration were normalised by hand. | gvf/E13.md |
| H.17 | closed | E12/E15/E16: in all four models with fields, value rides the fields at their source, and nothing looks like a late comparison. bcnv11 is reactive. | gvf/E12.md, E15.md, E16.md; E3 §5e |
| H.18 | open | Is bcnv11's objective-free axis component (0.40–0.44 of its energy) inert, as the DRC's is? | gvf/E15.md |
| H.19 | open | E14: the fitted readout at the agent's cell predicts choice but is not the decision. Next: pool over cells or ticks. | gvf/E14.md |
| H.20 | open | Tick-resolved patching (when, not just where), and interchange interventions for a formal factorisation test. | memory `gain-vs-factorised-study` |
| H.21 | open | Spatial half-distance is not the natural propagation measure for global-attention models (ViT, BC), and the ResNet's routing is not established. | gvf/E16.md; memory |
| M.1 | closed | The axis is findable in a memoriser. Only the drift-free, held-out behavioural write tells a learner from a memoriser. | `overfit/overfit-axis.txt` (ec95ae2); E3 §6 |
| M.2 | open | Memoriser follow-ups: the o1 sweep, more seeds, a DRC twin. The 3k rung finished and sits between 1k and 10k (held-out written/arms 2.8/1.6). | `results/overfit/overfit-axis.txt` |
| K.1 | closed (n=1) | Craftax stage 2 (ore fields): a one-knob axis on a competent base (cos −0.98, reliability 0.79, slopes ±23 against the expert's 20). | `craftax/value_axis.cxnv15c.*` (af26738); `docs/Craftax.md` |
| K.2 | closed (n=1) | Craftax stage 4 (crafting): an axis on `cxcraft15.s1t` (cos −0.862, reliability 0.59; arms and writes ±23 against ±20). It required a canonical expert and a table channel. | `craftax/value_or_gap.cxcraft15.s1t.txt` (0e700ec) |
| K.3 | open | **H1/H2 on Craftax stage 4.** Is the axis an input gain on the iron cue? Tests: embedding-row norms, brightness on the iron plane, restricted writes. Does raised iron value propagate to instrumental steps? The price-probe tables exist but are unanalysed. | `craftax/price.cxcraft15.s1t*.txt`; memory `craftax-stream` |
| K.4 | open | Craftax replicates. The from-scratch `cxcraft15t.s1` has only an o0 table; there is no second seed for either stage. | `craftax/value_axis.cxcraft15t.s1.o0.txt` |
| K.5 | open | Craftax stage 3 (local view + history): deferred. | `docs/Craftax.md` |
| K.6 | closed | Bring `docs/Craftax.md` up to date: it stopped before stage 4's table-channel result and quoted the superseded (2.0, 0.5) / 30-action design. Updated 2026-10-09. | `docs/Craftax.md`; commits b921329, 30fb705 |

## Retargeting cost: the metric's first pilot (route model, 2026-10-10)

The owner's EDL spec, in its simplest form: block-prequential code with doubling blocks, a refit from the base before each block, and early stopping on a fixed V. Bits are those of the expert's own moves; L* = 0 on perfect mazes. Code: `goalmisgen/offline/retarget.py`, `scripts/studies/retarget/`, `scripts/pods/retarget_chain.sh`.

| id | status | thread | evidence |
|---|---|---|---|
| RC.1 | closed (pilot) | Does the retargeting cost separate an agent with a dial from one without? Yes, with a constructed dial: the goal-vector agent `bcgv11.s1` retargeted by a learner that moves only its goal vector pays at most 1–65 bits over its own oracle dial setting, against a floor of about 3. `bcnv11` (which has the Experiment 2 value axis) pays 0.35k–5.3k bits over its null arm, growing with the size of the behaviour change and still rising at n = 16k. The memoriser fails the competence condition (RC.8) and decides nothing. | report https://claude.ai/artifact/WZkGyjxZG4V2FtHyq6CY3Q; `retarget/pilot.txt` |
| RC.2 | closed | Which stage is noisiest? None matters much: data order, learner seed, \|V\| (256 vs 1024), lr (1e-5 vs 3e-5) and the base seed each move one arm's cost by −5% to +9%, data order most. EDL is the exception: n times the test set's sampling error is about ±330 bits, and it came out negative on 5 of 8 arms, so it is unusable at this n. | `retarget/pilot.txt` |
| RC.3 | open | **The learner drifts a converged base.** A fresh Adam at the arms' 1e-4 raises the loss on levels the goal leaves alone from 0.27 to 0.48 bits per route at t = 256. The pilot uses 3e-5 (chosen on τ = 4.5, which no arm uses). Early stopping then keeps fits to tens or hundreds of steps. Is a full-weight Adam the right A? RC.7 says the learner matters as much as the agent. | `retarget/pick_lr.bcnv11.s1.txt` |
| RC.4 | open | Principled form of the code, deferred until the pilot showed what breaks. Tied actions: code only what the goal family determines (a class code), or label with θ₀ conditioned on the goal (charges collateral drift). The competence condition (RC.8). The early-stopping side channel (K log₂ C bits) against the floor. The reference to subtract (RC.9). | spec notes 2026-10-10 |
| RC.5 | open | The comparison family (the d₁ deadline, L(0)-matched) that separates a built-in dial from general plasticity. Not in the pilot. | spec §2 |
| RC.6 | closed | Build an agent with a dial by construction (the positive control). `bcgv11.s1`: the bcnv11 recipe on the o1 sweep's 25 sets (τ 1–19) with a learned goal vector per set at SEP. Swapping vectors sets its decoded exchange rate monotonically (τ 1 → 2.7, 10 → 12.0, 19 → 23.6, a +2 bias like bcnv11's); midpoints interpolate; one component holds 61% of the vectors' variance. It is less competent on fresh mazes than bcnv11 (93% reach against 98%). | `retarget/goal_vector_check.bcgv11.s1.txt`; `scripts/pods/goal_vector_base.sh` |
| RC.7 | open | **Full-weight SGD is slow to use a dial that exists.** On `bcgv11.s1` the dial-only learner reaches the oracle level within 4 demonstrations; the unrestricted learner takes 2–4 thousand, and its excess over the oracle peaks at up to 461 bits. So the metric measures (agent, learner) together, as the spec's framing says. Is bcnv11's high cost a missing dial, or a dial its learner cannot find? A restricted learner along the Experiment 2 axis would tell. | report §2, A7 |
| RC.8 | open | **Competence must be a gate, not something the null arm subtracts away.** The memoriser starts at 92–116 bits per route on fresh mazes at every τ, so its cost is relearning navigation; its SDL minus null is negative below τ = 10 (it already prefers the nearer objective). A threshold on the null arm's L(0) belongs in the spec. | report A4 |
| RC.9 | open | **Null-arm subtraction is biased by route composition.** Routes shorten as τ falls (8.9 tokens at 0.5, 10.2 at the null, 11.8 at 18.5), and residual error depends on which routes an arm holds. Per-route and per-token subtraction disagree by about ±0.6k bits for bcnv11 (tolerable) and ±2.5k for bcgv11 (not). The agent's own best dial setting (an oracle) removes the bias where one exists; nothing equivalent exists yet for an agent without a known dial. | report core 1 bars, A7; `retarget/oracle_dial.bcgv11.s1.txt` |
| RC.10 | open | Tension to resolve: Experiment 2 finds a one-dimensional value axis in bcnv11, yet retargeting it costs 20–100% of its ceiling. Does the axis need the drift component (M.1) that fine-tuning must also pay for? Or do fine-tuning arms move far more than the axis does? | E3 §6; RC.1 |
| RC.11 | closed | **Does the positive control still pass with a competent base?** `bcgv11.s1` saw only 50k mazes (1.03 bits per route on T, 93% reach), so its null arm kept improving. `bcgvr11.s1` is the same recipe on bcnv11's 900k mazes, each re-solved at one of the same 25 rates drawn at random. Its base matches bcnv11: 0.335 bits per route at τ = 10 (bcnv11.s1 0.332), 97% reach on the holdout, and the null arm is flat (0.335 → 0.335 over 16k demos). The dial is monotone in the middle with a +1.5–2 bias (τ 10 → 11.6, 4 → 5.5, 16 → 20.2) but saturates at both ends (τ 1–1.8 all decode ≈ 2.5, 17.6–19 ≈ 22–26). The dial-only learner still pays ≤ 89 bits over the oracle dial on every arm (−718 at τ 0.5, where it extrapolates past the trained vectors). Full-weight SGD at 3e-5 is slower than on bcgv11: 71–951 bits over the oracle, and at τ 2.5, 12.5, 14.5 and 18.5 it is still 0.01–0.04 bits per route above the oracle at n = 16k (it reaches it at ~4k for τ 0.5 and 6.5). So RC.7 holds more strongly once the base is competent: the metric measures the learner as much as the agent. | `retarget/goal_vector_check.bcgvr11.s1.txt`, `retarget/oracle_dial.bcgvr11.s1.txt`, `retarget/pilot.txt`; `scripts/pods/goal_vector_random.sh` |
| RC.12 | open | **Is bcnv11's cost a missing dial or one SGD cannot find?** (RC.7, RC.10.) Retarget it with the learner restricted to the coefficients of fixed weight-space directions from its o1@1k sweep: the Experiment 2 axis, the axis refitted without arms near the goal, axis + drift, the top 1/2/4 principal directions of the arm diffs, and 1-D/4-D random controls; `oracle_subspace.py` gives each 1-D direction's best setting. Runs were launched 2026-10-12; their results are not recorded here yet. | `RUNS.toml` `offline/subspaces/*`; `scripts/studies/retarget/{make_subspace,oracle_subspace}.py`; `retarget_chain.sh` phase `subspace` |
| RC.13 | open | **Does a dial built in survive memorisation?** `bcgvm11`: the bcgv11 recipe on 1k mazes per goal, as the memoriser bcov11.n1k saw. It has a goal slot by construction and no navigation beyond its mazes, so it separates having a dial from passing the competence gate (RC.8). Results not recorded here yet. | `RUNS.toml` `offline/runs/bcgvm11.s*`; `N=1k scripts/pods/goal_vector_base.sh` |
| RC.14 | open | **Requential coding as the code** (Qiu et al. 2026, arXiv 2607.11883): the student starts at the base and trains on draws a teacher selects, paying the cumulative teacher-student KL plus about 6-7 bits per call. Questions: does it remove the null-arm floor and route-composition bias (RC.9), since shared residual error is never charged? Does a restricted teacher (goal slot, value axis) let a full-weight student find a dial (RC.7)? Does teacher drift (RC.3) dominate the cost unless the teacher is smoothed? First run 2026-10-26 on the positive control `bcgvr11.s1`, 8 goals x 3 teachers (goal vector only; all weights; all weights from an EMA), 64 calls of 4,096 routes, student all weights at 3e-5. The teachers rank as predicted at every goal: goal vector < EMA < all weights (17k / 23k / 49k bits at tau 2.5), and smoothing halves the all-weights cost. The goal-vector teacher finds the oracle dial (best T 0.333 vs the oracle's 0.329 at tau 2.5, 0.345 vs 0.346 at 6.5, 0.332 vs 0.333 at 18.5). **No usable cost yet, for three reasons.** (1) A floor: at the agent's own goal the goal-vector teacher stays at the base (0.335 to 0.341) but the student, training on samples, random-walks to 0.37-0.39 and pays about 200 bits a call, 13.5k bits for no change of goal; the paper codes against an EMA of the student, which is not implemented here. (2) The all-weights teacher damages the agent: 16 passes over D with no stopping take it from 0.335 to 0.443 at its own goal, best at 4k-65k samples; a teacher is never sent, so it may be early-stopped on V for free. (3) The finish line charged per token is wrong: the oracle dial sits at 0.33-0.35 bits per route at every goal it can reach, so residual error is per route; the per-token line (0.295 at tau 2.5) is out of reach even for a perfect dial. Overhead is 1.2-1.5k bits a run (7-10%). | `retarget/requential_control.bcgvr11.s1.txt`; records on the volume under `offline/results/retarget-requential/`; `scripts/pods/requential_control.sh` |

## Not yet claimed (from CLAIMS.md), and framing

| id | status | thread | evidence |
|---|---|---|---|
| NY.1 | dropped | "An internal representation moves before misgeneralisation appears in behaviour." No longer the project's question; the one direct test (E1.6) was negative. | 2026-10-09 framing |
| NY.2 | dropped | Whether an agent that can read value off its input stores it at all. Dropped with R.4. **Still a limitation:** every value-axis agent hides its values. | R.4 |
| NY.3 | closed | Is the writable threshold a property of recurrent planning? No: the feedforward ResNet and ViT have the axis, and so does the reactive bcnv11. | G.2; E3 §5e |
| NY.4 | open | Does the axis survive a shift in the distance distribution? Nothing has tested it. | `c9358c5:CLAIMS.md` |
| NY.5 | open | **Formalise the setup as a metric runnable on LLMs.** This is the owner's current work. M.1 says the metric must include a drift-free held-out behavioural write. | E3 §6, §8 |

## Infrastructure and housekeeping

| id | status | thread | evidence |
|---|---|---|---|
| IN.1 | closed | Held-out evaluation before 2026-08-22 shared maze layouts with training. Mazes are now held out (`envs/splits.py`). Old pools are left alone, and no claim depends on solving unseen mazes. | `c9358c5:CLAIMS.md` "Known limitations"; c9358c5 |
| IN.2 | open | **Pool-size split trap.** `split_by_layout` thresholds are fractions of the pool, so splits are global only at a fixed pool size. Any cross-pool comparison needs a set held out from both. | `overfit/overfit-axis.txt`; acd2f84 |
| IN.3 | closed | `CLAUDE.md` and `README.md` carried the old misgeneralisation framing, and `CLAUDE.md` said the DRC trains at γ=0.995 (all banked RL bases are γ=1). Both rewritten 2026-10-09. | `CLAUDE.md`; G.1 |
| IN.4 | closed | Merged remote branches deleted 2026-10-09 (with `origin/craftax`) after `main` was pushed. | `git branch -r` |
| IN.5 | closed | Strip gymnasium warnings from the results tables. | e2eeffd |
