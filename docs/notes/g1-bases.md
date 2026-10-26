# The γ=1 base campaign (2026-08-29/30)

Session notes, copied into the repo on 2026-10-26 from working memory, trimmed
to the decisions and findings behind the banked g1 bases (THREADS G.1, G.3,
G.4). Why each run exists is in `RUNS.toml`.

## Rules the campaign kept

- Training budgets are uniform per family: DRC 150M, ResNet 150M, ViT 250M. A
  mixed budget within a family was vetoed.
- Base health is judged by the greedy decode (reach and threshold), not by
  training curves. `charts/0/returned_avg_episode_success` is a *sampled*
  training statistic; the argmax evaluation is `rho100/00_episode_successes`.
  Two ViTs (`vit11nvg1.s1234`, `s5678`) looked converged on the sampled
  statistic and never were under argmax (0.53 / 0.50 throughout); their arms and
  measures are excluded.
- The healthy-checkpoint picker's 0.95 bar assumes DRC/ResNet-style plateaus
  near 1.0. ViTs plateau at 0.90-0.95, so `vit11nvg1.s5678` defaulted to its
  collapsed final checkpoint and its `BASE.json` was repaired by hand (bar 0.85).
  A per-architecture bar is needed if more ViTs are trained.

## Findings

- γ=1 enlarges the value targets: the unnormalised ResNet value head diverged
  (loss 1e8-1e9 against 0.2 at γ=0.995) and was rerolled under a stated gate
  (value loss > 1e3 at 10M → kill and reroll a new seed). The DRC is indifferent
  to γ; its takeoff at 100-140M is normal at either γ.
- Every banked base undershoots the task's 10-step threshold (6.9-8.0 for DRC
  and ResNet, 5.2 for the ViT), so the undershoot survives undiscounting
  (THREADS UR.9).
- Argmax-corrected in-budget convergence: DRC 2 of 6 seeds, ResNet 3 of 6
  (including rescues), ViT 1 of 3. The ViT's failure is an argmax-against-sampled
  gap, not unfinished takeoff, so tail-LR extensions did not fix it
  (`vit11nvg1.s1234e` plateaued at 56-66%).

## The DRC budget decision (THREADS G.4, open)

The DRC third base `s3456e` is an extension of a 150M dud (+37.5M at the tail
learning rate 4e-6). Only s1234 and s9012 converged in 150M; the other seeds
(s5678, s7890 among them) were mid-takeoff at the 150M bell. Options set out at the time:

1. move the DRC family to a uniform 250M budget and retrain all three
   (recommended; ~24 GPU-h; keeps the no-mixed-budgets rule);
2. ship with two DRC bases and a "2/6 seeds converged within budget" caption;
3. more 150M rerolls (p ≈ 1/3 each).

The figures went ahead with `s3456e` as the third base; option 1 was never run.

## The ViT learning-rate probe (THREADS G.3)

`results/g1/vit-lr-probe.txt`: (peak, floor) pairs of the 250M schedule, killed
at ~70-80M and decoded greedily. The peak is the lever and the optimum is
interior. The probe measured the wrong thing: `episode_successes` rewards
reaching *a* goal. The tuned-LR trio (`vit11nvg1lr.*`, peak 1e-3, floor 1e-4)
reaches goals but runs to the nearest one (τ ≈ 1.5-2.0, ~2 steps of arm range
where the task demands 16), so the ViT base rate is about 1 in 6. A peak 1e-3 /
floor 4e-6 run finished 250M and was killed at the arms stage; its checkpoints
are on the volume, and whether it values goals is untested.
