# Repainting the objectives: swap and blend (observation only, no weight edits)

2026-09-17, main session. `scripts/studies/gain-vs-factorised/colour_swap.py`, `scripts/studies/gain-vs-factorised/colour_blend.py`
(both new). 4,096 held-out levels per condition, test split, greedy
policy. JSON on the volume at `figures-data/gain-vs-factorised/{swap,blend}-*.json`.

Nothing here edits weights or uses a fitted direction. The observation's two
colour planes are permuted or mixed, which is on distribution at the endpoints
and a linear mix between them.

## 1. Swap, decomposed into direction and strength

A colour's signature has a length and a direction, and the two colours differ in
both, so exchanging the planes moves both at once. Scaling each plane by the
other's signature length separates them.

| | base tau | swap | direction only | strength only | direction share | strength share |
|---|---|---|---|---|---|---|
| DRC s1234 | 7.06 | -7.37 | 2.37 | -2.26 | 33% | 65% |
| DRC s9012 | 7.60 | -7.71 | 3.18 | -3.39 | 29% | 72% |
| DRC s3456e | 5.28 | -5.73 | 0.43 | -0.65 | 44% | 54% |
| ResNet s1234 | 7.48 | -7.43 | -5.90 | 6.00 | 90% | 10% |
| ViT s9012 | 5.24 | -5.34 | -5.34 | 5.18 | 100% | 0.6% |

Shares are each condition's shift as a fraction of the full swap's shift; they
sum to 98-101% everywhere, so the two carriers are separable and additive.
Reach and route optimality are unchanged in every condition.

**The swap itself is not evidence** — colour is the only thing distinguishing the
objectives, so exchanging the planes must invert the threshold. It is kept as a
symmetry control: |tau_swap| - tau_base is 0.31 / 0.11 / 0.45 / -0.05 / 0.10
steps, i.e. under 6%, so there is no colour bias independent of value.

**The decomposition is the result:** the DRC keeps about two thirds of what
distinguishes its colours in the signature's LENGTH, the ResNet a tenth, the ViT
none. The ViT's row restates its base length ratio (1.008) rather than testing
anything, so it is nearly vacuous; the DRC and ResNet rows are not.

NB this measures how the BASE code distinguishes the colours, which is a
different question from what fine-tuning moves. The ResNet is
direction-dominated here (90%) while its fitted axis's activation change is
gain-shaped (E7: R^2 0.27 vs 0.04). Both can be true; keep them apart.

## 2. Blend: graded or categorical?

Both objectives are mixed towards each other: A = (1-a)*colour0 + a*colour1 and
B = (1-a)*colour1 + a*colour0. a = 0 is the base, a = 0.5 makes them identical,
a = 1 is the swap. Width is the logistic scale in steps (bigger = less decisive).

DRC s1234:

| a | 0.00 | 0.20 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| tau | 7.06 | 4.56 | 1.71 | 0.92 | -0.03 | -0.92 | -1.78 | -4.56 | -7.37 |
| width | 1.44 | 1.42 | 1.46 | 1.47 | 1.52 | 1.47 | 1.41 | 1.39 | 1.61 |
| reached | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| route | 94.8 | 95.0 | 95.3 | 95.1 | 95.2 | 94.9 | 94.9 | 95.3 | 94.7 |

ResNet s1234: tau 7.48 / 6.06 / 2.61 / 0.03 / -2.58 / -6.06 / -7.43 at
a = 0 / .2 / .4 / .5 / .6 / .8 / 1; width 0.92 -> 1.26 at the midpoint -> 0.89;
reached 99.1% -> 96.9% -> 98.9%.

ViT s9012: tau 5.24 / 4.67 / 2.13 / -0.00 / -2.56 / -4.89 / -5.34; width flat
2.83-3.15; reached flat 94.2-94.9%.

**All three are graded, not categorical.** A colour classifier feeding a value
lookup predicts a flat stretch then a jump, with confusion at the midpoint:
lower reach and a blown-up choice curve. Instead tau ramps smoothly through
zero and the agents stay fully competent facing two identical objects, where
"go to the nearer one" is the correct policy.

Shape differences worth a sentence: the DRC's ramp is near-linear in a (about
-1.4 steps per 0.1); the ViT's is S-shaped, steep in the middle and flat at the
ends; the ResNet shows the only hint of confusion, width +40% and reach down
2.2 points at the midpoint, recovering by a = 0.8.

The zero crossing at a = 0.5 is forced by symmetry under either account, so it
is not evidence -- but it lands at 0.500 (tau -0.03, 0.03, -0.00), which is a
second confirmation that colour bias is negligible.

## Caveats

- Blends between the endpoints are off distribution: training never contained a
  square at 0.851 or a two-colour blend. Competence holding throughout is
  reassurance, not proof.
- One base per architecture for the blend (DRC s1234, ResNet s1234, ViT s9012);
  the swap has three DRC bases.
- BC is not covered: the route model needs the offline decode path.
