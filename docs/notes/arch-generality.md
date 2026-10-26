# Architecture generality (2026-08-20/21, γ=0.995)

Session notes, copied into the repo on 2026-10-26 from working memory and the
two stream hand-offs (`~/projects/gm-worktrees/{arch-swap,offline-bc}.HANDOFF.md`,
not in the repo), trimmed to findings and open follow-ups. Tables:
`results/arch-swap/`, `results/offline-bc/`; `results/README.md` has the detail.

## What the two streams found

Two parallel streams asked whether Experiment 2's value axis (one writable
threshold knob, cos(axis_0, axis_1) ≈ −1) is a DRC quirk. It is not: it
reproduces on

- an RL ResNet (the healthy 130M checkpoint; cos −0.50 raw, reliability ~0.36),
- an RL ViT (cos −0.45, reliability ~0.39, base only 79% competent),
- a 0.8M-parameter prefix-LM trained by imitation (`bcnv11`, 3 seeds; cos −0.98
  raw, reliability 0.96), the cleanest of the three.

Experiment-1-style by-products (THREADS E1.4-E1.6): the misgeneralisation gap
reproduces on the feedforward nets, but "the proxy is learnt after competence"
(E1.5) does not. The ResNet's gap is present as soon as it is competent (both by
~4-5M steps), and the route model acquires colour before competence (its gap
opens when reach is 0.15-0.38). The plan probe does not lead behaviour on any
architecture (E1.6).

All of this is superseded as evidence for the axis by the γ=1 bases (THREADS
G.2), but it is the record for E1.5.

## Follow-ups noted and not done (THREADS AG.2, AG.3)

- **ResNet training instability**: `resnet11novalue.s1234` collapsed between
  130M and 140M under the DRC's IMPALA hyperparameters; the γ=1 ResNets collapse
  late too (AG.2).
- **A fully-causal route model.** The route model is a prefix-LM: bidirectional
  attention over the maze tokens and SEP, causal over actions. A fully-causal
  variant was noted as a follow-up in the offline-BC hand-off and never built
  (AG.3). The motivating observation: the per-cell residual holds the route only
  weakly (AUC 0.74 against 0.60 untrained), and much of the route is computed at
  the causal action positions, which the probe does not read.
- **Discrete cell tokens** instead of `Dense(observation channels)` plus learned
  row and column embeddings (AG.3; noted in memory only, no design written).
- CI bands and past-the-flip arms for the route model's fig-13 analogue (later
  covered by the g1 figures and the k arms).
- The offline-BC hand-off also asked whether the route-model arms should be
  redone on demos from the `valid` split to remove a 5.7% layout overlap; this
  was never done (see THREADS IN.6 for the larger overlap found later).
