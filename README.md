# Mechanistic goal-directedness in maze agents

4YP research code. Agents trained to choose between objectives of different
value at different distances are behaviourally goal-directed: they trade value
against distance at a consistent exchange rate. This project asks what structure
in the weights implements that trade-off, how well it can be understood, and
whether it can be exploited for AI-safety-relevant decisions. The main finding
so far is a **value axis**: one direction in weight space that sets the agent's
threshold on the distance gap. It can be written into a frozen agent, and it
appears in every base that learned the trade-off, in all four architectures.
Most of the code is the testbed: a multi-objective maze, four architectures
trained on it, and a Craftax stage.

## What is here

- **The environment** (`goalmisgen/envs`): symbolic multi-objective mazes,
  levels generated into fingerprinted datasets, splits held out by maze layout,
  with ground truth (BFS distances, the optimal target) in `info`, never in the
  observation.
- **Four architectures**: a DRC (Deep Repeated ConvLSTM), a ResNet and a ViT
  trained by RL, and a prefix-LM trained by behavioural cloning on expert routes
  (the "route model", `goalmisgen/offline`). The RL agents train with far.ai's
  cleanba stack, used **unmodified** as a pinned submodule; our environment and
  networks plug in by subclassing (`goalmisgen/configs`, `goalmisgen/nets`).
- **Craftax** (`goalmisgen/craftax`): Craftax-Classic used as an engine, with
  maze levels rendered into it (stage 2) and a crafting choice (stage 4), for
  the route model.
- **Analysis and edits** (`goalmisgen/analysis`, `goalmisgen/edits`): probes,
  value-axis fits, restricted and dosed writes, field decompositions, and a
  search for the cheapest weight edit that sets a threshold.
- **Experiments and scripts**: `experiments/` holds the numbered experiments,
  each standalone with its question in the docstring; `scripts/` holds drivers,
  pod chains, reports and one-off measurements.
- **Results and figures**: `results/` keeps the verbatim output of every
  measurement (`INDEX.md` says which script wrote each file, `README.md` why);
  `figures/` builds the write-up and blog figures from them.
- **Data**: datasets, runs and bulk figure data live on a RunPod network volume
  (`/workspace/data`), not in the repo. `RUNS.toml` says why each run exists;
  `MANIFEST.md` lists what is there.

## Layout

| path | contents |
|---|---|
| `THREADS.md` | The ledger of questions: open, closed, dropped, and where each answer lives |
| `docs/` | Experiment write-ups (`Experiment1-3.md`), `UtilityRule.md`, `Craftax.md` |
| `CLAUDE.md` | Working rules and hard-won facts about the stack (read before changing code) |
| `goalmisgen/` | The package: environment, configs, networks, offline route model, Craftax, analysis, edits |
| `experiments/` | Numbered experiment scripts, and the record of the early value-axis questions |
| `scripts/` | Dataset generation, sweep drivers, pod chains, reports |
| `results/` | Raw measurement output; `INDEX.md` (generated) and `README.md` |
| `figures/` | Figure scripts, rendered figures and blog fragments; see `figures/README.md` |
| `RUNS.toml`, `MANIFEST.md` | Why each run on the data volume exists; what is there (generated) |
| `archive/` | Superseded results and volume provenance, each with a README |
| `tests/` | Unit and integration tests |
| `third_party/train-learned-planner` | far.ai's JAX/cleanba training stack, as a pinned submodule |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/rahul-marchand/GoalMisgeneralisation
cd GoalMisgeneralisation
git submodule update --init third_party/train-learned-planner
git -C third_party/train-learned-planner submodule update --init third_party/gym-sokoban
uv sync
```

The submodules are initialised one level at a time rather than with
`--recurse-submodules`: `train-learned-planner` has its own `envpool` submodule,
a large C++ build used only for fast Sokoban, which we do not need (it is
imported lazily) and leave uninitialised. `gym-sokoban` *is* required, because
`cleanba.environments` imports it at module scope.

On a GPU machine, add the CUDA build of JAX:

```sh
uv sync --extra gpu
```

## Tests

```sh
uv run pytest
```

## Where to read next

`THREADS.md` for what is known and open, then `docs/Experiment3.md` for the
current state of the mechanism work. `docs/Experiment1.md` and
`docs/Experiment2.md` record the earlier stages as they stood at the time.
