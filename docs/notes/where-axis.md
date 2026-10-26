# Where the axis sits: design notes (2026-09-07)

Session notes, copied into the repo on 2026-10-26 from working memory, trimmed
to the measures, findings and the one unbuilt figure (THREADS W.1-W.3). The
write-up is `docs/Experiment3.md` §2; tables `results/where-axis/`; figures
`figures/fragments/where-axis.html` and `axis-dose.html` (blog figures 5 and 6).

## The share measure

Per-module share of the *sweep-agreeing* axis, −⟨v̂₀,M, v̂₁,M⟩ / −⟨v̂₀, v̂₁⟩.
The two sweeps' fitted axes carry independent fine-tuning noise, so the product
is unbiased; it sums to 100%, and slightly negative modules are drawn at zero.
The raw ‖v̂‖² share is dominated by noise in the large flatten dense. Module
grouping is `goalmisgen.analysis.layers.module_of / forward_order /
module_groups`. The RL nets' `Dense_0` / `dense_list_0` is cleanba's
flatten→256 readout, not an embedding.

## Restricted writes and doses

`scripts/measure/restricted_writes.py` (RL) and `restricted_writes_bc.py`
(route model): per module, written alone, alone rescaled to the axis's length,
and left out of a full write, both signs at ±0.30, plus a largest-k cumulative
ranked by the other sweep. `--doses` writes each slice at ×0.5 to ×8 of its own
size.

- **DRC** (3 bases): the input conv (1,184 weights) alone gives 42% of the full
  write, 92% rescaled (×2.4; sufficient), and 59% remains when it is left out
  (not necessary). Effects are roughly additive over the early layers (85-102%).
  The flatten dense holds 12% of the axis's length and none of its effect;
  recurrent weights ~0. The input-conv slice is >99% on the two colour planes:
  raising a colour's value amplifies its plane's kernel (cos +0.64 to +0.80)
  and attenuates the other's (−0.48 to −0.65), a gain on how loudly each colour
  is seen.
- **ResNet**: input conv and stages 0-3; nothing from stage 4 on; dense zero.
- **ViT**: no module carries it as written; block 0's MLP does when rescaled.
- **Route model**: spread over the MLPs.
- **Dose**: every tensor with an effect is linear (slope 1) up to the full
  write. The DRC input conv is superlinear past ×2; ResNet stage 1 and the route
  model's block 1 MLP saturate at ×8; tensors with no effect stay at the noise
  floor at every dose.

## Not built (THREADS W.3)

A plane-decomposition mini-figure: the DRC input-conv slice split by input
plane (walls, agent, colour 0, colour 1), showing the two-colour gain directly.
`scripts/studies/where-axis/conv0_planes.py` writes its data,
`figures/data/restricted/conv0_planes.json` (untracked; written in the pod's checkout on
the volume, `/workspace/GoalMisgeneralisation`).

The figure design that was kept: one panel per architecture, three bars per
tensor (share of parameters, share of the axis's length, share of its effect =
written-alone shift / full write); necessity is one sentence in the text and
linearity is the dose figure.
