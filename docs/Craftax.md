# Craftax: the value axis in a second engine

*Design and status. Started 2026-09-15 on branch `craftax`, merged cc7d9f4.
Updated 2026-10-09. Open questions are `THREADS.md` K.3-K.5.*

## Question

On the mazes, fine-tuning a hidden-value agent onto shifted objective values
moves its weights along one writable direction, the value axis, in all four
architectures. How it is implemented differs. In the DRC it is a two-plane gain
on the colour input. In the route model, which is the architecture used here,
it sits in the MLPs, and no input-level edit that keeps the model competent
reproduces it (gain-vs-factorised E7, E9). The 12.7M route model fuses value
into its distance fields; the 0.8M one is reactive (E15).

Craftax asks two things of the same pipeline in a richer, recognised engine:

1. Does the axis appear in a prefix-LM cloned from a utility-optimal expert
   inside Craftax-Classic (17 actions, solid ores, a crafting tree)?
2. When an objective sits behind a crafting chain, does raising its value raise
   the worth of the instrumental steps that unlock it, none of which carry its
   cue? An axis that only rescales the objective's input cue cannot do that.

## Engine, not task

`craftax==1.6.1` is used unmodified: we build its states and step them with
`craftax_step`. Ours are the worlds, the experts, the observations and the
bridge, all in `goalmisgen/craftax/`. The route-model pipeline is shared with
the maze through the `Demonstrations` protocol
(`goalmisgen/offline/demonstrations.py`). `CLAUDE.md` lists the rules that keep
the boundary clean.

## Stages

| stage | world | observation | expert | model |
|---|---|---|---|---|
| 2 | 15×15 ore field: bedrock border and scattered bedrock (density 0.3), coal and diamond at hidden values (1.0, 0.5), player holding the pickaxes | the maze's channel stack, full map | canonical maze route + DO, `value - 0.05 × actions` | the bcnv11 recipe (4 layers, d=128), `n_actions=17`, open-loop |
| 3 (deferred) | as 2 | Craftax's own 7×9 local view + inventory, with history | full-state planner, or explore-then-plan | – |
| 4 | 15×15 field: bedrock walls (density 0.2), six trees, four stone deposits, iron and coal at hidden values (1.1, 0.5), bare-handed player | current map, facing, inventory, and a channel for a placed table | greedy canonical crafting plan, then the ore | d=512, 4 layers (12.6M), receding-horizon |

**Stage 2** is the maze task executed by the engine: every decoded route is
replayed by the real engine, so a wrong action id or a misplaced DO shows there.
Ores are solid, so the expert's last move only turns the player to face the ore
and DO mines it; the turn is dropped after a straight approach (cost `d + 1`
after a corner, `d` after a straight run).

**Stage 4** puts each ore behind a chain. Coal needs a wood pickaxe: three wood,
two for the table and one for the pickaxe. Iron also needs a stone pickaxe: one
more wood, one stone, and a return to the table. The expert is a canonical
greedy policy: nearest tree first (ties by cell order), table on the third tree
cut, wood pickaxe; for iron, the nearer of the fourth tree and the nearest stone,
then the other, back beside the table, stone pickaxe; then the ore. Legs are
shortest paths with one fixed move preference (`craftax/routes.py`), so every
decision is inferable from the map. The choice is still `value - 0.05 × plan
cost`. Values (1.1, 0.5) put the threshold at 12 actions, the median extra cost
of the iron plan over the coal plan (p25/50/75 = 8/12.5/19), so the expert takes
iron on about half the fields and the ±0.45 arms (thresholds 3 to 21) sweep most
of the distribution.

The stage-4 model sees the state as it is now and emits one action per forward
pass, with the engine in the loop (`craftax/closed_loop.py`). It is trained on
every state along an expert route paired with the remaining route (suffix
training). `craftax/dagger.py` can add states the policy visits, labelled by
the planner from that state.

**Arms and analysis** are the maze route model's: `goalmisgen.design` sweeps,
1k steps at constant lr 3e-5 on demonstrations at the shifted values, values
still hidden; `027` and `028` fit and compare the axes. Chains:
`scripts/pods/craftax_chain.sh` (stage 2), `scripts/pods/craft_chain.sh` (stage 4).
Diagnostics: `scripts/studies/craftax/craft_diagnose.py` (where decoded routes leave the
expert's) and `scripts/studies/craftax/craft_price_probe.py` (episodes started with wood, stone
or a pickaxe already in hand).

## Mistakes on the way

- **Stone walls.** A wood pickaxe mines stone, so the first stage-2 base
  tunnelled in 42% of its routes. Walls are now bedrock (the furnace block,
  inert in these recipes) in both stages (533ae85).
- **A non-canonical expert.** The first stage-4 expert ranked tree orders over
  permutations of static distances. The model could not tell which tree came
  first and hedged on the first move: 3% reach at 85% token accuracy, on
  training and held-out fields alike. The canonical greedy expert replaced it
  (b921329), which also moved the values from (2.0, 0.5) and a 30-action
  threshold to (1.1, 0.5) and 12 actions.
- **Open-loop decoding.** The open-loop prefix-LM loses its route at the first
  bump. Stage 4 became receding-horizon (e2ce780).
- **No table channel.** A placed table rendered as grass, so the state was not
  Markov once it was down. The coal chain never returns to the table and still
  worked; the iron chain failed exactly at the walk back. DAgger moved the
  failure, which is how it was found. `cxcraft15.s1t` is `cxcraft15.s1` widened
  by one zero-initialised input row (`scripts/studies/craftax/craft_widen_input.py`) and trained
  on (30fb705).
- **Under-training** on stage 2: the overnight base reached an ore on 54% of
  fields at 20k steps. Training to 60k, not the expert's tie-break, closed the
  gap.

## Results (one seed per stage)

| base | stage | reach | cos(axis₀, axis₁) | reliability | arm / written / expert slope |
|---|---|---|---|---|---|
| `cxnv15.s1` (20k steps) | 2 | 54% | −0.99 | 0.87 | 13 / 10.5 / 20 |
| `cxnv15c.s1` (canonical expert, 60k) | 2 | 89.5% | −0.98 | 0.79 | ±23 / +21.7, −19.3 / ±20 |
| `cxcraft15.s1` (no table channel, 60k) | 4 | 63% | −0.86 | 0.59 | +18.5 / +16.7 / +20 (o0) |
| `cxcraft15.s1t` (table channel, +20k) | 4 | 98% | −0.86 | 0.59 | +23.3, −23.7 / +22.6, −22.9 / ±20 |

Slopes are actions of threshold per unit value. On stage 2, norm-matched random
directions move the threshold by at most 0.3. Tables: `results/value_axis.cx*`,
`results/value_or_gap.cx*`. `cxcraft15.s1t`'s axis statistics are close to
`cxcraft15.s1`'s because it is that model, warm-started.

- **Stage 2.** A one-knob, writable axis, on the competent control as on the
  half-trained base. The competent base's arms and writes track the expert;
  the half-trained one's slopes are about two thirds of it.
- **Stage 4.** With the table channel the model reaches 98% and matches the
  expert's route on 70% of held-out fields; arms and held-out writes both track
  the expert's ±20. The axis is there even on the base without the table
  channel.
- **Price probe** (`results/price.*`). The 5k-step `s1t` checkpoint's iron/coal
  exchange rate on the expert's own states is 11.1 actions against the expert's
  12. Off that manifold, an extra ingredient in hand acts as a switch rather
  than a price: one wood or one stone sends it to iron on 94-100% of fields
  where the expert goes on 55-68%. On the ±0.45 arms the start-of-episode iron
  rate is 0.07 against 0.73. Whether a raised iron value propagates to the
  instrumental steps is not analysed beyond these tables.

## Not done

- **The mechanism behind question 2 is untested.** No restricted writes,
  brightness sweeps, input gains or field decompositions have been run on a
  Craftax base, so it is not known whether the stage-4 axis acts on the iron
  cue or downstream of it (K.3).
- **Replicates.** One seed per stage; the from-scratch `cxcraft15t.s1` has only
  an o0 sweep (arm slope +23.2, written +22.0) (K.4).
- **Stage 3** is deferred (K.5).
