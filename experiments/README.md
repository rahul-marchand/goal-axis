# Experiments

Runnable scripts, numbered in the order they were written. Each is standalone and
prints its own reasoning; the docstring at the top of a file says what question
it asks and what would refute it.

| script | asks |
|---|---|
| `001_maze_repro.py` | Trains an RL agent on multi-objective mazes (`--net` for DRC, ResNet or ViT; `--gamma`, `--hide-values`). Every RL base came from here. |
| `002_measure_proxy.py` | How much does the agent follow colour rather than value, swept across correlations? |
| `003_probe_plan.py` | Is the route the agent will walk already linearly decodable before it moves? |
| `005_outcome_keyed_distance.py` | Are distances represented, keyed by which outcome they belong to? |
| `006_psychometric.py` | What exchange rate between value and distance did the agent actually learn? |
| `007_steer_distance.py` | Does writing a distance into the state move the decision? |
| `008_steer_at_the_head.py` | The same, at the readout rather than in the recurrence. |
| `009_value_representation.py` | Has an objective's value travelled to where the decision is made? |
| `010_choice_direction.py` | Does a direction built from the agent's own choices steer it? |
| `011_target_probe.py` | Is the objective it will take already decided before it moves? |
| `012_rewrite_the_plan.py` | Write a whole alternative route into the cell state; does it walk it? |
| `013_value_axis.py` | Fine-tune onto a new objective value, and keep the weight change. |
| `014_value_axis_analysis.py` | Do those changes share a direction, and can it be written to? |
| `015_value_or_gap.py` | Is the knob a value, or the gap between two values? |
| `016_three_objective_values.py` | With three objectives, does moving two values equal the sum of moving each? |
| `017_where_the_axis_lives.py` | How much of a fine-tune is shared, and where does the axis sit? |
| `018_which_channels.py` | Which gates and channels, and do two independent sweeps agree? |
| `019_restrict_the_axis.py` | Keep only the enriched channels — what still reads, what still writes? |
| `020_are_they_value_channels.py` | Do those channels carry the decision in the agent's own activations? |
| `021_own_task.py` | Is each arm actually good at the task it was fine-tuned on? |
| `022_axis_ladder.py` | When in a base's training does the value axis appear, and when does it become one axis? |
| `023_train_bc.py` | Trains the route model (a prefix-LM) on expert demonstrations. Every route-model base and arm came from here. |
| `024_bc_proxy.py` | Does a route model trained on correlated demonstrations follow the colour? (twin of `002`) |
| `025_bc_probe_plan.py` | Is the route in the residual stream before the first action token? (twin of `003`) |
| `026_bc_early_warning.py` | Does the probe notice the colour before the decoded routes show it? |
| `027_bc_value_axis.py` | Is an objective's worth a direction in the route model's weights? (twin of `014`) |
| `028_bc_value_or_gap.py` | One knob or two, for the route model? (twin of `015`) |
| `029_axis_shape.py` | How many value axes are there, and is writing one in one place enough? |
| `030_bc_rewrite_the_plan.py` | Write a route into the residual stream; does the model walk it? (twin of `012`) |
| `031_bc_outcome_keyed_distance.py` | Does the route model hold a distance to both objectives, or only to its target? (twin of `005`) |
| `032_bc_write_the_distance.py` | Write a distance into the end-of-input token; does the decision move by that much? |

`004` does not exist; the numbering has a gap rather than a renumbering, so
references in older results files still resolve. Work since September is mostly
in `scripts/` rather than here; `results/INDEX.md` names the script behind each
table.

---

## The value-axis line: what was asked, what came back

A record of the questions in the order they were asked, because each one exists
only because of how the previous one turned out, and that chain is not
recoverable from the code. `results/README.md` says what each output file is;
`MANIFEST.md` says what is on the volume; this says why any of it happened.

All of it was done in August 2026 on `novalue11.s1234`, a DRC trained at
γ=0.995, with experiments `013`-`020`. Where later work showed a statement here
to be wrong, it is corrected in place with a pointer. The current state is in
`THREADS.md` and `docs/Experiment3.md`.

Mistakes are included. Three of the results below were wrong the first time and
the reasons are more useful than the corrections.

### The question underneath

Do agents develop modular goal representations — something like a value per
objective, separable from the machinery that compares them — without being
trained to? `novalue11` is the sharp case: it has **no value channel at all**.
Colour is the only cue to what an objective is worth, so the values are learned
constants rather than inputs. Nothing forces it to represent "1.0" and "0.5"; it
could compile the whole comparison into a threshold on the difference in
distances and throw the values away.

That also makes it unprobeable in the usual way. With constants there is no
per-episode value to regress against, so the correlational toolkit is
unavailable by construction.

### 013, 014 — manufacture the variance, fit an axis

**Question.** Is there a direction in weight space that sets what an objective is
worth?

**Method.** Fine-tune the same agent onto a grid of values for colour 1, and fit
`diff = drift + offset × axis` across the arms.

**Answer: yes, behaviourally.** An axis fitted *without* an arm reproduces that
arm's exchange rate — five of six within 0.6 steps against a range of 9.5 — at
100% reach throughout. Norm-matched random directions of the same magnitude do
nothing, and so does the null arm, which is a real full-size fine-tune with no
value change.

**But the weights themselves say nothing.** Individual diffs are not collinear,
held-out R² sits at zero, and the fitted axis is about a sixth signal. The
behavioural test works where the weight-space test fails because behaviour only
sees a small subspace, and the arm-specific movement is large in norm but
almost entirely outside it.

**Mistake.** The first fit was forced through the origin, on the reasoning that a
zero change must mean a zero diff. The null arm refutes that: at zero offset it
moved 14.41. Forcing through zero did not remove the shared component, it
absorbed it, and the axis then read back the same offset for every arm. Fitting
an intercept fixed it. `tests/test_weights.py` builds that exact situation.

### 015 — is it a value, or the gap?

**Question.** With two objectives the choice turns on a single difference, so "the
agent holds what colour 1 is worth" and "the agent holds a threshold on the
distance gap" predict the same policy. Which is it?

**Method.** Sweep *colour 0's* value over the same gaps and compare the two axes.
One shared knob means they are anti-parallel; two registers means they need not
be.

**Answer at the time: undecided.** Raw cosine −0.255 with reliability around
0.15, so the disattenuated figure landed outside the range a cosine can occupy.

**Mistake.** `--cross` was added to settle it causally and cannot. Raising colour
0 and lowering colour 1 by the same amount leave the same gap, so *both*
hypotheses predict the agreement it found — after the same file had already
argued that no behavioural test can separate them. The flag is kept, with its
docstring corrected, as the record of a test that does not do what it was built
to do.

### 016 and the three-objective grid — composition

**Question.** Three objectives make the choice depend on two independent
differences, so one scalar is not enough. Does moving two values equal the sum of
moving each?

**Method.** One axis per objective from single-value arms, then arms with *two*
values moved, held out of every fit.

**Answer: composition holds behaviourally.** The sum of two single-value axes
reproduces an arm neither was fitted on, in exchange rate *and* in choice
quality — including a drop to 83.1% optimal on the arm where the values are
squeezed together, which the composed edit reproduces at 83.0%.

**Rank alone would not have shown this.** An agent that solves a three-way choice
must depend on two differences, so two dimensions are close to forced by the
task. Composition is not forced, which is why it is the test.

Every weight-space number in that run is uninformative: reliability came out at
0.04 to 0.09. The grid was rerun wide (`threeobj2`) with doubled offsets.

### 017, 018 — what is moving, and where

**Every arm moves the same distance whatever it learned**: 14.74 with a spread of
0.36, the null arm included at 14.41. The size of a fine-tune is set by the
optimiser and the update count, not by how much there is to learn. About a third
of that movement in energy is a direction every arm shares, and it changes no
behaviour at all.

**The axis is not sparse.** Its largest tenth of a percent of parameters holds 4%
of its length against 1.3% for a gaussian — heavy-tailed, but a circuit would put
most of the length there. A single arm and the shared component have the same
tail, so even that is a property of fine-tuning rather than of the value.

**Channels do differ, repeatably.** In the first recurrent layer, channels 7 and 1
carry about 2.2 times their share, a secondary set follows at 1.2 to 1.5, and the
median channel sits below 1. No gate specialisation whatsoever: input, candidate,
forget and output all between 0.92 and 1.06. The channel preference replicates
across the colour-0 and colour-1 sweeps as well as two halves of one sweep
replicate each other.

**And the signed test settles 015: one knob.** Restricted to the channels that
carry the axis, with both sides fitted from the same number of arms,
`cos(axis_0, axis_1) = −1.00` within a few per cent at every subset and in both
gate convolutions. Raising one objective's value and raising the other's move the
same weights in exactly opposite directions. That is the right answer for a
two-objective task, where one scalar suffices.

**Mistake.** The first version of that comparison read six-arm fits against a
ceiling built from three-arm fits, which understates the ceiling and inflates the
correction. Matching the arm counts is what turned −1.7 into −1.00.

**Later.** "One knob" held up: the wide sweeps gave raw cosines of −0.76 to −0.84
at reliability ~0.8, and every γ=1 base in all four architectures sits at its
noise ceiling under a permutation null (`results/one-axis-null.txt`).

**Correction: these channel analyses never looked at the input convolution.**
`017` grouped it into a single encoder block, and `018`-`019` select ConvLSTM
kernels only. Module-by-module writes on the γ=1 DRCs later found the input
conv's 1,184 weights carry most of the effect: written alone and rescaled to the
axis's length they give 92% of the full write, almost all of it on the two
colour planes (`docs/Experiment3.md` §2). The channel findings below describe
the recurrent layers only.

### 019 — keep only the enriched channels

**Reading improves, writing degrades, monotonically, in the same table.**

Masking to 9,216 parameters — 0.6% of the network, 2.8% of the axis's length —
takes the held-out read-back from a slope of 0.03, which is the same number
returned for every arm, to 0.55 at r = 0.96. The offset an arm was trained at
becomes readable off a checkpoint the axis never saw.

Writing goes the other way: the full axis reaches 2.4 extra steps from a base of
7.7, those channels alone reach 6.8, and 5.1 once rescaled. Signal-to-noise
inside the mask governs what can be read; total behaviourally effective magnitude
governs what can be written, and those channels have the first without the
second.

**Layer 0 is special, and this is where that was established rather than assumed.**
Its top two channels move behaviour; layer 1's barely do; layer 2's do nothing at
all, 7.7 against a base of 7.7, even scaled up ninefold.

**Mistake.** The read-back was first computed in-sample and reproduced the true
offsets almost exactly — meaninglessly. The axis is a weighted sum of the same
arms being projected, each weighted by its own offset, so each arm's own noise
dominates its own projection and the offsets come back because the estimator put
them there. Held out, the full axis is flat.

### 020 — are they value channels?

**No.** Three tests, none using a fitted axis:

- **Ablation.** Zeroing channels 7 and 1 in the untouched agent gives 7.4 steps
  against random pairs at 7.6 to 7.9. Nothing.
- **Steering.** Non-monotonic in the amount added, and a random pair moves the
  exchange rate further, mostly by damaging the agent.
- **Probe.** Ranked by how well their activation predicts which objective is
  taken, channels 7 and 1 come **15th and 30th of 32**.

The channels that *do* predict the choice — 3, 19, 23 — are equally inert:
zeroed, they give 7.9 against an untouched 7.8, while random triples give 7.6 to
7.7.

So the channel enrichment is a real statistical fingerprint of where fine-tuning
writes, and it is not a mechanism. This is the Hase et al. point measured inside
a single agent with both sides on the table: gradient descent wrote where writing
was cheap, which is not where the quantity is held.

**Limit.** The gate convolution takes the observation as input on every tick, and
an intervention can only be applied between environment steps, so the three ticks
inside a step can rebuild whatever was zeroed. A null here shows no small channel
set is necessary *at that granularity*. Separating "distributed" from
"recomputed within the step" needs an intervention inside the tick loop.

### A practical finding about the method, since corrected

On this agent, split-half reliability fell with arm length: 0.228 at 750k
steps, 0.177 at 1.5M, 0.160 at 2.2M, 0.144 at 3M. The conclusion drawn at the
time, that behaviour converges by 750k and longer arms only add arm-specific
movement, **was wrong**. The arm-length ladder on the three γ=1 DRCs
(`results/len-*`, `docs/Experiment3.md` §3) found reliability flat from 400k to
3M, the fine-tuned slope still rising to about 1M steps (14 to 16-17 steps per
unit value, against the task's 20), and arms past parity reversing only at
0.75-1.4M. What grows with length is the drift's share of each arm (22-28% to
41-51%). Arm length is a parameter to report, not one to minimise.

Reliability scales as `r/(1−r) ∝ |u|²·Σ(offset − mean)²/σ²`, so widening the
offsets buys more than adding arms — `Σ` grows with the square of the offset and
only linearly with the count — but it is bounded by nonlinearity, which shows up
as composition degrading at ±0.2 and extrapolation failing outside the grid.

### What happened next

The open questions this section used to list (a second seed, the
three-objective agent's structure, a step-penalty specificity control) are
threads in `THREADS.md`, with their outcomes.
