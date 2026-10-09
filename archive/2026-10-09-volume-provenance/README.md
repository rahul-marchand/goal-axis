# Volume provenance — archived 2026-10-09

What sat loose at the top of the RunPod volume (`/workspace`) before the
2026-10-09 clean-up: the launch scripts, job files and logs of every pod chain
from 2026-08-04 to 2026-10-02, the old-layout `/workspace/results` and
`/workspace/logs`, and the 41 files of the `h1/` scratch directory not
duplicated in `figures/data/h1`. `listing.txt` is the archive's contents.

It is provenance, not code the repo depends on: how each run and figure was
launched, with what flags, on which pod. Six one-off analyses that produced
committed results were promoted to `scripts/` (`arm_norms`, `bc_angle`,
`colour_angle`, `bc_extrap_far`, `fit_axis_cache`, `conv0_planes`); the rest
(`decomp.py`, the prototype of `analysis/arm_length.py`; `stick.py`; E-series
debugging and report assemblers; `h1/`) are only here.

The `metrics.csv` and `NOTE.md` of the runs deleted from the volume that day
(dud and excluded bases, the ViT learning-rate probe, the legend control) are
kept outside the repo, in `~/projects/gm-archive/run-records-2026-10-09.tar.gz`
on the Surface, since they are 57 MB uncompressed.
