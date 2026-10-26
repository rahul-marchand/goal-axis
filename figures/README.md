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
| `scripts/studies/arch-swap/arch_swap_report.py`, `scripts/studies/experiment1/early_warning_report.py` | `fig_arch_swap_dynamics`, `fig_early_warning` | run metrics and `results/` |

`extract_value_axis.py` and `extract_wide_axis.py` turn results tables into the
JSON these read.

**File names do not match caption numbers.** Where each figure appears:

| file | used as |
|---|---|
| `fig1_task` | `docs/Experiment1.md` Figure 1 |
| `fig5_example_plan` | `docs/Experiment1.md` Figure 2 |
| `fig8_intervention` | `docs/Experiment1.md` Figure 3 |
| `fig2_rho_response` | `docs/Experiment1.md` Figure 4 |
| `fig3_margin` | `docs/Experiment1.md` Figure 5 |
| `fig4_dynamics` | `docs/Experiment1.md` Figure 6 |
| `fig9_no_value_task` | `docs/Experiment2.md` Figure 1 |
| `fig10_written_value` | `docs/Experiment2.md` Figure 2 |
| `fig11_ood_writes` | `docs/Experiment2.md` Figure 3 |
| `fig_distance_rule`, `fig_axis_dial`, `fig_gain_knob` | `docs/UtilityRule.md` parts 1, 2, 3 (unnumbered) |

No doc uses `fig6_distance_field`, `fig12_written_ood`, `fig13_wide_threshold`,
the five `fig_bc_*`, `fig_arch_swap_dynamics` or `fig_early_warning` (the last two
are named in `results/README.md`), nor any figure in `craftax/`.

## 2. Blog fragments (`src/` → `fragments/`)

Each blog figure is a self-contained HTML fragment with build-time SVG and no
runtime JavaScript (except `basin`, which has a slider).
`src/<name>/build_fig.py` renders `src/<name>/data.json` into
`fragments/<name>.html`, which is spliced into the post at `<!-- fig: name -->`.
Builders run with plain `python3`. Checked 2026-10-26: `basin`, `task-planning`,
`axis-dose`, `hierarchy` and the retarget report rebuild byte-identically;
`value-axis`, `value-axis-both`, `value-axis-far`, `scurves`, `where-axis` and
`arm-length` rebuild with differences only in the last digits of SVG
coordinates (float noise), so the committed fragments were left as published.

| fragment | `data.json` comes from |
|---|---|
| `value-axis`, `value-axis-both` | `src/value-axis/parse_g1.py` (the `results/g1/` tables and the route model's JSON) |
| `value-axis-far` | `src/value-axis/parse_far.py`, written into `src/value-axis/data.json` |
| `scurves` | `extract_scurves.py` (decode npz on the volume) |
| `where-axis` | `extract_where_axis.py` (restricted-write JSON) |
| `axis-dose` | `extract_axis_dose.py` (`--doses` JSON) |
| `arm-length` | `scripts/studies/arm-length/arm_length_report.py --out` |
| `basin` | `scripts/measure/basin_map.py`, injected into `src/basin/template.html` |
| `task-planning` | `scripts/measure/plan_heatmap.py`, `scripts/measure/bc_plan_heatmap.py` |
| `hierarchy` | none: `src/hierarchy/gen.py` draws a fixed diagram |
| `retarget-pilot` (a standalone report page, not a blog fragment) | `scripts/studies/retarget/retarget_report.py --data`; prose in `src/retarget-pilot/notes.json` |
| `model-table` | typeset by hand from `results/plan-probe/probe-plan-*` |

In the published post (https://rahulmarchand.com/posts/decision-threshold/, as
of 2026-10-26): `basin` is the unnumbered interactive demo at the top,
`task-planning` is figure 2, `value-axis` 3, `arm-length` 4, `where-axis` 5,
`axis-dose` 6, `scurves` 7, and `value-axis-far` the appendix-G figure.
`value-axis-both` and `hierarchy` are not in it.

**The two parsers in `src/value-axis/`** both assemble per-architecture,
per-sweep series of trained arms against axis writes. `parse_g1.py` builds
fig 3's panels (`xpanels`, `fit_window`), windowed to the fitted range.
`parse_far.py` builds the appendix's panels (`farpanels`), unwindowed, and
records the offset at which a written agent's reach collapses rather than
cutting it silently. Run in that order from an empty `data.json`, they
regenerate it exactly from `results/g1/` and `--bc`, a directory holding
`figures/data/bc/value_axis.bcnv11.*.json` plus the sidecars
`extrap_axisonly{,_far}.json` (untracked: the far one is written to
`figures-data/bc/` by `scripts/studies/value-axis/bc_extrap_far.py`; the near
one's producer, a pod job, was never committed). Rebuilt that way on 2026-10-26, which dropped keys no
builder read (`series`, `scurves`, `panels`, `base_values`, left from an earlier
design).

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
under `/workspace/data/figures-data/` (`figures-data/` for short). `figures/data/`
is gitignored: files tracked before the policy (84e80ca) stay tracked, new ones
are not added. The route model's arm copies once tracked under
`figures/data/bc/arms/` (576 files nothing read) were removed on 2026-10-26; the
arms themselves are on the volume.
