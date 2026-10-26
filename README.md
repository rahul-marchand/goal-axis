# A steerable decision threshold in maze agents

Research code for **[A Steerable Decision Threshold in a Planning Agent](https://rahulmarchand.com/posts/decision-threshold/)**.
Read the post first; this repo is its record.

Agents solve 11×11 mazes holding two goals of different value, which they
cannot see. The optimal policy accepts up to a fixed number of extra steps to
reach the richer goal: a decision threshold. In every agent that learned the
trade-off (a DRC, a ResNet and a ViT trained by RL, and a transformer trained by
behaviour cloning), that threshold is **a single direction in the weights**.
Adding the direction to a trained agent moves its threshold about as precisely
as fine-tuning on a different reward, and leaves its navigation intact.

**Follow-up (upcoming):** turning this into a scalable metric of mechanistic
goal-directedness: how many bits it takes to retarget an agent to a new goal.
That work continues in a new repository; the pilot that started it is recorded
here (`THREADS.md`, RC.*).

## Where to read

| | |
|---|---|
| [the blog post](https://rahulmarchand.com/posts/decision-threshold/) | The results and their limits |
| `THREADS.md` | Every question asked, its status, and where the evidence is, including corrections found after the post |
| `docs/` | Experiment write-ups as they stood at the time (`Experiment1-3.md`, `UtilityRule.md`, `Craftax.md`) and the notes `THREADS.md` cites |
| `results/`, `figures/` | Verbatim measurement output, one folder per study, and the figures built from it; each has a README |
| `CLAUDE.md` | How the code fits together and the traps it has hit; read before changing anything |

The repo began as a goal-misgeneralisation project, which is why the package is
called `goalmisgen`. Training uses far.ai's
[train-learned-planner](https://github.com/AlignmentResearch/train-learned-planner)
unmodified, as a submodule. Datasets and checkpoints live on a RunPod volume, not
in git (`RUNS.toml` says why each run exists, `MANIFEST.md` what is there).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/rahul-marchand/GoalMisgeneralisation
cd GoalMisgeneralisation
git submodule update --init third_party/train-learned-planner
git -C third_party/train-learned-planner submodule update --init third_party/gym-sokoban
uv sync              # add --extra gpu for CUDA JAX
uv run pytest
```

Initialise the submodules one level at a time as above: the planner's own
`envpool` submodule is a large C++ build this project does not need.
