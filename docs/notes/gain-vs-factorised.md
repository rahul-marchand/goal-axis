# Gain against factorised value: study notes (2026-09-15..17)

Session notes, copied into the repo on 2026-10-26 from working memory, trimmed
to the framing, cross-experiment findings and the proposals the per-experiment
reports do not hold. Each experiment's own report is
`results/gain-vs-factorised/E<N>.md`; the summary is `docs/Experiment3.md` §5;
status is THREADS H.1-H.21.

## The question

H1: the axis is a gain on colour input, so value is carried as colour-feature
strength inside the plan. H2: a separate value variable is combined late.
Operationally: the fraction of the axis's τ effect carried by gain-aligned
activation change, per layer, with brightness (plane scaling) as H1's positive
control. Later redrawn (E12) as: everything except a reactive policy
propagates, so the question is *what* propagates and *how* two objectives
combine: (A) a utility field with value at the source and competition,
(B) salience diffusion with value at the source and summation, (C) value-free
distance fields with value applied at a late comparison, (E) reactive.

## Cross-architecture findings not in a single report

- **Brightness across architectures** (from `results/input-gain/`):
  brightening colour 1 lowers τ only on the DRC (×3 → 0.0); the ResNet is flat;
  the ViT and the route model go the wrong way (ViT ×3 → 8.3 from 5.2; route
  model ×3 → 32 from 10.8, U-shaped). Dimming colour 0 lowers τ everywhere.
  Axis share in the input layer: DRC 17.6%, ResNet 1.1%, ViT 0.7%, route model
  0.00%. This led to the hypothesis E7 tested: value is salience in whatever
  coordinate the architecture preserves, magnitude where it survives (DRC,
  ResNet: no input normalisation) and direction where a pre-LayerNorm deletes it.
- **Five-model synthesis after E12-E16**: value rides the fields at their source
  wherever fields exist, and nothing looks like a late comparison.

  | model | γ fit | propagation (half-distance) | bridge corr | crossing vs τ |
  |---|---|---|---|---|
  | DRC | 0.63 | 3-4 → 6-11 cells over ticks | −0.66 | 8.53 vs 7.06 (s1234) |
  | ResNet | 0.52-0.91, non-monotone | 4 cells, never halves | −0.83 | 7.41 vs 7.48 |
  | ViT | 0.62-0.73 | 1 cell at every block | −0.59 | 7.45 vs 5.24 |
  | route d512l4 | 0.64 | 1 → 10 cells | −0.76 | 10.92 vs 10.72 |
  | route bcnv11 | 0.83-0.92 | 1 cell, flat | −0.07 to −0.23 | meaningless |

  In every model with fields, the axis's interaction share is below the
  source-change control's (ResNet 0.27 vs 0.41, ViT 0.21 vs 0.28, d512l4
  0.22-0.29 vs 0.43-0.45). Routing against blur: the ViT routes at depth (BFS
  0.65-0.73 vs grid 0.015), d512l4 does (0.99 vs 0.96), the ResNet is not
  established (grid wins at stage 8).

## Proposals not yet run

- **Positive control where value is factorised (H.6).** E5's legend DRC barely
  uses value (base τ 0.9, slope ≈ 1.6 steps per unit against the task's 20).
  Candidates, roughly cheapest first: a route-model twin with explicit value
  tokens in the prefix and colourless objectives (factorised by construction,
  hours not days); a route-model twin on the legend observation; a legend DRC
  with a wider value gap, or 250M at the normal learning rate; a global
  per-colour price channel; a successor-feature agent (ψ·w), the
  literature-backed option.
- **Bridge test (H.11).** Across an ordinary o1 sweep, the angle between the two
  colours' input-kernel signatures tracks trained value monotonically in all
  three RL architectures (`colour-signature-arms.md`). Does each arm's measured
  angle change predict that arm's measured τ at the rotation edits' angle→τ rate
  (ViT: 5.8° ≈ 11 steps)? If so, "the axis is the turn" becomes causal rather
  than correlational.
- **Programme handed over by the discussion fork (H.20)**, ranked: (1) an
  objective-additivity 2×2 decomposition with `leftover.py` (Δ_both against
  Δ_only0 + Δ_only1 per cell and tick; the interaction term says how factorised
  the two objective pathways are, then whether the value manipulation scales an
  objective's own contribution or the interaction; E12 later did a version of
  this); (2) tick-resolved patching, when and not just where; (3) interchange
  interventions for a formal factorisation test; (4) the blend sweep (done,
  `colour-swap-blend.md`).
- **Caveat on E16 (H.21).** The ViT and the route models have global attention,
  so a spatial half-distance is not the natural propagation measure for them,
  although d512l4's 1 → 10 cells shows the measure can detect propagation when
  it is there. The ResNet's routing is not established. ViT n=1.
