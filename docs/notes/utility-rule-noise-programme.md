# The utility rule's noise programme (route model, 2026-08-27/28)

Session notes, copied into the repo on 2026-10-26 from the working memory they
lived in, trimmed to findings, decisions and paths. This is the unwritten part 4
of `docs/UtilityRule.md` (THREADS UR.2). Tables are in `results/utility-rule/`;
arrays were in `figures/data/h1/` and are on the volume under `figures-data/`.
Scripts are in `scripts/studies/utility-rule/`.

## The question

Part 3 leaves a gauge: behaviour cannot tell a wobbling threshold ξ from a
misread gap ε (ξ ≈ −θ̄ε). The chosen next step was to read the model's own
distance-gap estimate from its activations; if choice is sharp in that estimate,
θ is constant and ξ lives in the distance computation.

Design decisions: the probe is trained on solver ground truth, never on choice
(circularity); within a fixed (d_rich, d_poor) cell the probe's variation is
pure residual, so residual → choice is the experiment; the reliability config
is chosen by gap R², not by the choice statistic. Yardsticks: hand-built
features explain ~1% of within-cell choice variance; cross-seed ξ correlation
≈ 0.3; ξ IQR ≈ 4.5 steps. Probes train on `train.rho100` and score on
`test.rho100` (level-disjoint; note that the two share most maze layouts,
THREADS IN.6).

## Findings

**Gap probes** (`goalmisgen/offline/gap_probe.py`, `probe_bc_distance.py`,
`gap-probe-*.txt`). Linear probes do read distances (gap R² ~0.5 at SEP by
block 4, emerging with depth; controls dead), but within a cell their residual
explains ~1% of choice variance (AUC 0.62-0.65, sign correct, 3 seeds).
Split-half reliability 0.86-0.89 with disattenuated choice correlation 0.08-0.09:
the stable residual is mostly systematic readout error, not the model's ε.

**Flip points** (probe-free ε; `flip_points_bc.py`, `flip-points-bcnv11.txt`).
In the informative band (gap 4-18) violations are 4.5-5.8% per decision against
12.6-16.1% under a no-stable-gap null; zero-violation levels 19-24% against ~0%.
(An earlier "55-59% clean" framing counted trivially clean censored levels and
was corrected.) Violations sit a median ~2.7 steps from the flip, so each edit
perturbs the per-level reading by ~2-3 steps. flip − gap median ≈ −0.1, IQR
3.5-3.9; cross-seed correlation +0.37 to +0.41. One threshold per fixed model
stands, but ε is not invariant across edits.

**Probe against flip** (`probe_flip_bc.py`, `probe-flip-*.txt`). Within-cell
correlation of the true-gap probe residual with the flip residual is
+0.18/+0.24/+0.20 at SEP block 4 (3-6% of variance). A flip-trained probe
predicts like the true-trained one (correlation 0.87-0.94): no extra linear ε
direction.

**DRC contrast** (`decode_h1_drc.py`, `decode-h1-drc.*`). The γ=0.995 DRCs cross
at 7.6-8.2 with pooled width 1.2-1.3 steps against bcnv11's 4.4-5.9: the DRC has
almost no ξ to explain, and it under-pursues. The noise programme therefore
lives on the route model. Not yet run: the same decode on the ResNet and ViT
(THREADS UR.7).

**Steering the decoded gap direction: null three times** (`goalmisgen/offline/steer.py`,
`steer_flip_bc.py`, `steer-flip-*`, `steer-prefix-*`; THREADS UR.3). ±8-step
calibrated writes at SEP depth 3 move the crossing by ~0 (slopes −0.006 to +0.002
against a predicted −1); 0.3-2% of band levels flip. The prefix-wide write is
also null. A uniform write across positions cancels in any readout that takes
spatial differences, and the gap is a difference, so the next candidate is an
**anti-symmetric per-cell write**: push the distance field up near the rich
objective and down near the poor one, as the DRC experiments 007/008 did
(THREADS UR.6, not run). The d512l4 run is null too (crossing pinned at 10.51 at
every dose, 2/5998 flips): reading is not using, at both scales.

**Distance fields** (`distance_field_bc.py`, `DistanceToAgent` in
`goalmisgen/analysis/targets.py`, `distance-field-*.txt`). bcnv11 barely has
fields: distance-to-agent partial r 0.19-0.22, d→rich 0.37-0.43, d→poor
0.18-0.20, and hard-cell R² negative (−0.08 to −0.5) where Manhattan distance
fails. DRC reference: partial 0.70, hard R² +0.25.

**Capacity co-collapse** (`distance-field-sc11.*`, `decode-sc11.*`; THREADS
UR.4, SC.3). Across the scaling twins, field hard-cell R² / threshold width:
d128l4 (0.8M) −0.10 / 4.93; d256l4 (3.2M) +0.35 / 1.37; d512l4 (12.7M) +0.57 /
0.60; d512l16 (50.5M) +0.64 / 0.50. The 1.45M DRC (+0.25 / 1.2) sits on the
same trend. ξ dies as distance fields become veridical, over 63× in parameters.
The August scaling campaign saw "no change" because it measured the axis knob
and `chose_optimal`, which are scale-invariant or saturated, never width or
activations. Transformers stay calibrated (crossing ~10.2-10.5).

**Margin decompositions** (`goalmisgen/offline/margins.py`, `margin_swap_bc.py`,
`margin-swap-*`, `margin-fork-*`). An exact variance decomposition of
first-action margins along colour-swap and constructed-start orbits (no probes).
37% / 30% (bcnv11.s1 / d512l4) of within-cell margin variance inverts under
swap, of which 76% / 88% varies with the start. The first-action bridge to
choice is weak (AUC 0.81), so the margin was re-read at the route's divergence
point (`--read fork`): within-cell AUC 0.79 / 0.947. T (location-difference
keyed, "distance") and S (maze-symmetric, threshold side) are orthogonal within
cells and carry choice-relevant noise in comparable measure; the joint logistic
gives ε:σ 1.33:1 / 1.66:1 (2.32:1 for d512l4 after a per-bin reanalysis).

**Link-assumption test.** Regressing the behavioural flip residual on (T, S)
gives β_S/β_T = +0.69 in both models. S is decision-real but at 69% of T's
fidelity, or the flip dial couples to σ sub-unitarily; the instruments cannot
separate the two. Fidelity-corrected: ε carries ~50-70% of choice-relevant
noise (48% bcnv11, 62% d512l4), σ 30-50%, and ε's share rises with capacity.

## State of the claim

"ξ = distance misestimates" is strong but correlational. For: the capacity
co-collapse (6 models, no exceptions), flip-point structure at every scale, the
sign and replication of bcnv11's probe residual, ξ as a cross-seed level
property. Against: every causal write failed; the representation the model
uses is not the linearly read one. Open (THREADS UR.5): the composition of S,
the origin of the 0.69 constant, and a probit-latent formalisation of the
shares. Every margin results file prints two permanent caveats: the gauge (a
fractional distance misread is equivalent to a per-maze price error), and that
T is location-difference keyed only broadly.
