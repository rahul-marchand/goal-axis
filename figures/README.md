# Figures

Three generations of figures live here, each with its own scripts and inputs.
None of them types a number in by hand: every figure is rebuilt from measured
data.

## 1. Write-up figures (matplotlib, August 2026)

`make_*.py` at this level draw the PNG/PDF pairs beside them (`fig1_task` …
`fig13_wide_threshold`, `fig_bc_*`, `fig_distance_rule`, `fig_axis_dial`,
`fig_gain_knob`) used by `docs/Experiment1.md`, `docs/Experiment2.md` and
`docs/UtilityRule.md`. Run with `uv run python figures/<script>.py`.

| script | figures | reads |
|---|---|---|
| `make_figures.py` | `fig1`-`fig12` | `data/*.json` and `data/*.csv` (experiment JSON, training metrics) |
| `make_wide_figures.py` | `fig13_wide_threshold` | `data/wide_value_axis.json` |
| `make_bc_figures.py`, `make_bc_fig13.py` | `fig_bc_*` | `data/bc/` |
| `make_distance_rule.py`, `make_axis_dial.py`, `make_gain_knob.py` | `UtilityRule.md` parts 1-3 | `data/h1/` |
| `scripts/arch_swap_report.py`, `scripts/early_warning_report.py` | `fig_arch_swap_dynamics`, `fig_early_warning` | run metrics and `results/` |

`extract_value_axis.py` and `extract_wide_axis.py` turn results tables into the
JSON these read.

## 2. Blog fragments (`src/` → `fragments/`)

Each blog figure is a self-contained HTML fragment with build-time SVG and no
runtime JavaScript (except `basin`, which has a slider).
`src/<name>/build_fig.py` renders `src/<name>/data.json` into
`fragments/<name>.html`, which is spliced into the post at `<!-- fig: name -->`.
Builders run with plain `python3` and every fragment rebuilds byte-identically.

| fragment | `data.json` comes from |
|---|---|
| `value-axis`, `value-axis-both` | `src/value-axis/parse_g1.py` (the `g1-*` tables and the route model's JSON) |
| `value-axis-far` | `src/value-axis/parse_far.py`, written into `src/value-axis/data.json` |
| `scurves` | `extract_scurves.py` (decode npz on the volume) |
| `where-axis` | `extract_where_axis.py` (restricted-write JSON) |
| `axis-dose` | `extract_axis_dose.py` (`--doses` JSON) |
| `arm-length` | `scripts/arm_length_report.py --out` |
| `basin` | `scripts/basin_map.py`, injected into `src/basin/template.html` |
| `task-planning` | `scripts/plan_heatmap.py`, `scripts/bc_plan_heatmap.py` |
| `hierarchy` | none: `src/hierarchy/gen.py` draws a fixed diagram |
| `model-table` | typeset by hand from `results/probe-plan-*` |

**The two parsers in `src/value-axis/`** both assemble per-architecture,
per-sweep series of trained arms against axis writes. `parse_g1.py` builds
fig 3's panels, windowed to the fitted range. `parse_far.py` builds the
appendix's panels, unwindowed, and records the offset at which a written
agent's reach collapses rather than cutting it silently. Both regenerate the
committed `data.json` exactly from the committed results.

**Builders deliberately do not share chart primitives.** The owner hand-tunes
each figure; a shared module would couple them. Duplicate a primitive rather
than edit another figure's builder.

`basin_standalone.html` and `valueaxis_far_standalone.html` are standalone
previews of two fragments, as published.

## 3. Craftax (`craftax/`)

PNG/PDF figures for `docs/Craftax.md` and the write-up, from what the Craftax
chains wrote into `data/craftax/`: `make_craftax_figures.py` (one figure per
claim, both stages), `make_craftax_axis_figure.py` (the value axis across
tasks, beside the maze route model), `make_craft_fig1.py` with
`craft_fig1_data.py` (the crafting task from a recorded rollout), and
`plot_craftax_examples.py` (example fields with the expert's route).

## Data policy

The repo keeps code, write-ups, `results/` tables and rendered figures. Figure
source data (npz and JSON) is measured on the pod and kept on the data volume
under `/workspace/data/figures-data`. `figures/data/` is gitignored: files
tracked before the policy (84e80ca) stay tracked, new ones are not added.
