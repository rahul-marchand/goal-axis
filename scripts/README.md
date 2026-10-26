# Scripts

Everything here is run from the repository root, e.g.
`uv run python scripts/measure/input_gain.py ...`. `experiments/` holds the
numbered experiments, each asking one question in its docstring; `scripts/`
holds the rest: the plumbing that makes datasets and arms, the measurement
tools several studies share, the scripts that belong to a single study, and
the shell chains that ran them on pods. The docstring at the top of every
script says what it asks and how to call it.

Until 2026-10-10 this was one flat directory of 124 files. It is now grouped by
role, so that a returning reader can tell the plumbing from the measurements,
and a study's one-off scripts from the tools other studies rely on. The old ->
new path of every file is in
[`archive/2026-10-10-reorganisation/scripts.tsv`](../archive/2026-10-10-reorganisation/scripts.tsv).
Results tables written before the move record the old path in their headers;
that is the path as it was run, and is left alone.

## Layout

| folder | role | what goes here |
|---|---|---|
| `data/` | datasets and bookkeeping | Anything that makes or indexes data rather than measuring an agent: level and demonstration datasets, the volume's naming migration, `MANIFEST.md` and `results/INDEX.md`. |
| `train/` | making agents | Drivers that train bases or fine-tune arms (or list the arms a chain should train). The training entry points themselves are `experiments/001` (RL) and `023` (route model). |
| `measure/` | shared measurement tools | Scripts whose output feeds more than one study, or that answer a generic question about any run. If a second study starts calling a study's script, move it here. |
| `studies/<study>/` | one study's scripts | Measurements and report writers that exist for one study, including its one-off report generators. Named like the study's folder in `results/` where it has one. |
| `pods/` | pod chains | The shell chains that ran campaigns on RunPod, and `launchers/`, the one-shot launchers. Records of how runs were launched as much as tools; see below. |

`scripts/` has no `__init__.py` anywhere; tests import a script as a
namespace package, `from scripts.data import migrate_volume`, with the repo
root on the path (`pyproject.toml`). Scripts never import each other: shared
code lives in `goalmisgen/`.

## `data/`

| script | what it does |
|---|---|
| `generate_levels.py` | Pre-generate a level dataset (see `CLAUDE.md`, "Level datasets"). |
| `generate_ore_fields.py` | Pre-generate an ore-field level dataset for the Craftax task. |
| `generate_demos.py` | Demonstrate a split of a level dataset at one correlation (route-model training data). |
| `generate_craftax_demos.py` | The same for an ore-field dataset, as Craftax routes (stage 2). |
| `generate_craft_demos.py` | Demonstrate a range of the crafting pool (stage 4). |
| `make_demo_subset.py` | The first N demonstrations of a set, optionally excluding another pool's (the overfit study). |
| `make_goal_split.py` | Split a demonstration set between several goals, each level re-solved at one goal drawn at random (the goal-vector agent `bcgvr11`, `THREADS.md` RC.11). |
| `resplit_by_layout.py` | Re-split an existing level dataset so train and test share no maze. |
| `migrate_volume.py` | Move the data volume onto the naming scheme in `goalmisgen/volume.py`. |
| `manifest.py` | Index every dataset and trained agent on the volume into `MANIFEST.md`. |
| `arm_eval_overlap.py` | Count evaluation levels that also sit in an arm's or a base's training split, by level and by layout (`THREADS.md` IN.2, IN.6). |
| `results_index.py` | Index every table under `results/` (recursively) by the script and commit that produced it, into `results/INDEX.md`. |

## `train/`

| script | what it does |
|---|---|
| `value_axis_sweep.py` | Fine-tune one RL agent onto the wide value grid (`goalmisgen/design.py`). |
| `base_ladder.py` | Run a value sweep from earlier checkpoints of one agent (the base-checkpoint ladder). |
| `frozen_arm.py` | An ordinary `013` value arm with some modules frozen. |
| `value_axis_arms.py` | List the route model's value-axis arms, one per line, for shell loops. |
| `scaling_arms.py` | The same for the width/depth campaign. |
| `pick_arm_lr.py` | Which fine-tuning learning rate to give one cell of the width/depth grid. |

## `measure/`

| script | what it does |
|---|---|
| `arm_thresholds.py`, `bc_arm_thresholds.py` | What one family of arms does, without fitting an axis (the k/x tables in `results/g1/`, the arm-length ladder). |
| `restricted_writes.py`, `restricted_writes_bc.py` | Write one module's slice of the value axis and read the behaviour; `--doses` for the dose curves. |
| `input_gain.py`, `input_gain_bc.py` | Threshold against each colour plane's input gain: is the axis a brightness knob? |
| `cheapest_edit.py` | The cheapest edit that moves the exchange rate, and whether it is the value axis. Home of the cheapest-edit study, and used by most gain-vs-factorised experiments (E4, E7, E9-E11). |
| `basin_map.py`, `fit_axis_cache.py`, `written_scurves.py` | Fit and cache an axis, roll it out from every cell, and dump per-gap choice curves (blog figures `basin`, `scurves`). |
| `plan_heatmap.py`, `bc_plan_heatmap.py` | Per-cell plan-probe scores for drawing (blog figure `task-planning`). |
| `competence.py` | Is an agent worth running an experiment around? Exits non-zero below the bar. |
| `optimum.py` | Mean optimal utility of the levels an evaluation actually scores. |
| `summarise_run.py`, `training_curve.py`, `convergence.py` | Has an RL run / a route model stopped improving, and what was its correlation gap? |

## `studies/`

| folder | study | scripts |
|---|---|---|
| `experiment1/` | Experiment 1 (γ=0.995 DRC): the plan, the proxy | `distance_field_map.py` (the distance-field pilot's figure data), `early_warning_report.py` (E1.6: did the probe move before behaviour?) |
| `arch-swap/` | Experiment 1 on a ResNet and a ViT | `arch_swap_report.py` |
| `value-axis/` | The value-axis figures across architectures | `bc_extrap_far.py` (the route model's far writes for the appendix), `bc_value_axis_table.py` (the route model's o-sweep summary) |
| `arm-length/` | Fine-tune length (`docs/Experiment3.md` §3) | `arm_length_ladder.py`, `arm_length_report.py` |
| `one-axis/` | One knob or two, against a null | `one_axis_null.py`, `one_axis_replicate_ceiling.py` |
| `where-axis/` | Where in the weights the axis acts (§2) | `conv0_planes.py`; the writes themselves are `measure/restricted_writes*.py` |
| `far-axis/` | A fresh axis past parity | `far_axis_fit.py`, `far_axis_fit_bc.py` |
| `cheapest-edit/` | The cheapest edit (§4) | `cheapest_edit_report.py`, `edit_compare.py`, `edit_overfit_check.py`, `fork_margin_diagnostic.py`; the search is `measure/cheapest_edit.py` |
| `gain-vs-factorised/` | E1-E16 (`results/gain-vs-factorised/`) | see below |
| `utility-rule/` | The route model's utility rule and the noise programme (`docs/UtilityRule.md`) | see below |
| `overfit/` | Does an axis need a base that learned the task? (§6) | `overfit_gap.py`, `overfit_axis_report.py` |
| `scaling/` | The width/depth campaign | `shape_probe.py` |
| `retarget/` | Retargeting cost, the metric's pilot (`THREADS.md` RC); the metric is the package `goalmisgen/retarget/`, the maze and route-model parts `goalmisgen/offline/retarget.py` | `retarget_cost.py` (one prequential code), `pick_lr.py` (the learner's rate), `goal_vector_check.py` and `oracle_dial.py` (the positive control's dial and its best setting), `oracle_subspace.py` and `make_subspace.py` (restricted learners along weight-space directions), `retarget_report.py` (table, warning lights and figure data), `diagnose.py` (every run's warning lights and a standard figure per run); chains `pods/retarget_chain.sh`, `pods/goal_vector_base.sh` (`bcgv11`) and `pods/goal_vector_random.sh` (`bcgvr11`, RC.11) |
| `craftax/` | Craftax stages 2 and 4 (`docs/Craftax.md`) | `craftax_report.py`, `craft_diagnose.py`, `craft_price_probe.py`, `craft_dagger.py`, `craft_widen_input.py` |

**`gain-vs-factorised/`**, by experiment (each `E<N>.md` names its code):

| experiment | scripts |
|---|---|
| E1, E8, E10 | `gain_alignment.py`, `control_writes.py`, `leftover_patch.py` |
| E2 | `plan_invariance.py` |
| E3 | `separability_decode.py`, `separability_report.py` |
| E4 | `edit_collateral.py`, `e4_summary.py` (report writer) |
| E6, E8 | `axis_far_dose.py`, `axis_far_dose_report.py`, `censored_tau.py` |
| E7, E9 | `direction_profile.py` |
| E11 | `e11_summary.py`, `e11_report.py` (report writers), `token_budget.py` |
| E12, E15, E16 | `field_decomp.py`, `route_field_decomp.py`, `feedforward_field_decomp.py` |
| E13 | `e13_position_patch.py` |
| E14 | `readout_fit.py`, `e14_report.py` (report writer) |
| colour signature, swap/blend | `arm_norms.py`, `colour_angle.py`, `bc_angle.py`, `colour_swap.py`, `colour_blend.py` |

**`utility-rule/`**: the per-level decodes everything else reads
(`decode_h1.py`, `decode_h1_drc.py`, `decode_grid.py`, `decode_written.py`,
`decode_masked.py`); parts 1-3 of `docs/UtilityRule.md`
(`utility_rule_report.py`, `h2_recheck.py`, `arm_threshold_shape.py`,
`gain_knob_report.py`, `gain_knob_checks.py`); and the unwritten part 4: flip
points (`flip_points_bc.py`, `flip_points_dir.py`), the gap probe
(`probe_bc_distance.py`, `probe_flip_bc.py`), steering it (`steer_flip_bc.py`),
margins (`margin_swap_bc.py`) and distance fields (`distance_field_bc.py`).

## `pods/`

The chains that ran each campaign on RunPod: `campaign.sh` (the γ=0.995
robustness campaign), `value_axis_grid.sh` and `value_axis_report.sh` (the
first value grid), `three_objective*.sh`, `arm_length_curve.sh`,
`early_warning.sh`, `arch_swap*.sh`, `offline_bc_pod.sh`,
`offline_value_axis_{pod,chain}.sh`, `scaling_pod.sh`, `overfit_axis_{pod,chain}.sh`,
`legend_chain.sh` (gain-vs-factorised E5), `craftax_chain.sh` and
`craft_chain.sh`, and `setup_cloud.sh` for a fresh instance. `launchers/` holds
the one-shot `run_*.sh` launchers of the utility-rule programme and the blog's
plan probes.

They are kept as the record of how each run was launched. Their script paths
were updated mechanically to this layout (and each chain still `cd`s to the
repo root first), but none has been rerun since the move: treat one as a
starting point to read before launching, not as a tested command. The
launchers hardcode `/workspace/GoalMisgeneralisation`.
