# Path maps of the 2026-10-10 reorganisation

On 2026-10-10 two flat directories were grouped so they can be navigated by eye:

- `scripts/` (124 files) by role, into `data/`, `train/`, `measure/`,
  `studies/<study>/` and `pods/`; see `scripts/README.md`.
- `results/` (687 tables outside `gain-vs-factorised/`) by study, one folder
  each; see `results/README.md`.

Files were moved with `git mv`. Nothing else in them changed, except paths that
point at other moved files; results tables were not edited at all.

`scripts.tsv` and `results.tsv` map each old path to its new one, one file per
line. Use them to follow a path recorded before the move: in a results table's
header, a pod log on the volume, a commit message or a memory note.
