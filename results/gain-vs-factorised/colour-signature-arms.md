# The colour's input signature across an arm sweep (weights only)

Measured 2026-09-17 from the main session, on the e10 pod, CPU, no rollouts.
Scripts (session scratchpad, not in the repo): `colour_angle.py`, `arm_norms.py`.

`K_c` is the input kernel's slice reading colour c's plane — for the DRC and
ResNet a 3x3x32 = 288-vector, for the ViT the 64-wide embedding row. Colour 0 is
the rich objective (value 1.0), colour 1 the swept one (base 0.5). Each arm is an
ordinary 400k o1 arm, trained by changing the REWARD; nothing here is an edit.

## Bases

| base | \|K0\| | \|K1\| | ratio | angle |
|---|---|---|---|---|
| DRC s1234 | 17.18 | 8.97 | 1.914 | 31.5 deg |
| DRC s9012 | 22.41 | 11.59 | 1.933 | 30.4 deg |
| DRC s3456e | 23.49 | 14.90 | 1.577 | 23.5 deg |
| ResNet s1234 | 2.755 | 2.135 | 1.291 | 77.3 deg |
| ResNet s2468 | 3.159 | 2.542 | 1.243 | 69.0 deg |
| ViT s9012 | 3.438 | 3.410 | **1.008** | 34.6 deg |

The DRC writes the rich colour's signature ~1.6-1.9x longer than the poor one's
(against a 2:1 value ratio). The ViT's are the same length to 1%.

## Arms: what fine-tuning moves

DRC s1234 (`ratio` = \|K0\|/\|K1\|):

| trained v1 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 null | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| \|K1\| | 8.669 | 8.709 | 8.780 | 8.843 | 8.961 | 9.187 | 9.411 | 9.652 | 9.931 |
| ratio | 2.006 | 1.992 | 1.975 | 1.955 | 1.922 | 1.867 | 1.813 | 1.751 | 1.682 |
| angle | 35.49 | 34.40 | 33.66 | 32.61 | 31.64 | 30.14 | 28.84 | 27.64 | 27.05 |

ViT s9012:

| trained v1 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 null | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.011 | 1.012 | 1.012 | 1.011 | 1.001 | 1.001 | 1.004 | 0.996 | 0.993 |
| angle | 35.96 | 35.92 | 35.34 | 34.96 | 34.66 | 34.48 | 34.17 | 33.54 | 33.36 |

ResNet s1234: ratio 1.336 -> 1.220 and angle 79.0 -> 73.8 deg over the same
range, monotone apart from small wobbles at the low end.

Route model bcnv11.s1 (`cell_in`, 4 x 128, so a signature is one 128-row;
measured over all 38 arms - the o0 and o1 sweeps and the colour-keyed k-arms):
base ratio 0.993, angle 85.50 deg; every arm 0.993-0.994 and 85.48-85.50 deg.
The lengths move in the fourth decimal. **The BC input code does not move at
all.** Its two colours start near-orthogonal (cos 0.078) and equal in length.

## What it says

- **Angle tracks value in all three RL architectures**, monotonically, and in
  the ViT's case across all nine arms. Movement over the sweep: DRC 8.4 deg,
  ResNet 5.2 deg, ViT 2.6 deg -- and BC **0.02 deg**, i.e. nothing.
- **Length tracks value only in the two unnormalised networks** (DRC -16%,
  ResNet -8.7% over the sweep). The ViT's changes by 1.8% and crosses 1, i.e.
  not at all.
- So the fine-tuned change to the input code is a **stretch and a turn together**
  where magnitude survives, and a **turn alone** where it does not. That is the
  same split the edit searches found (E7: gain reach 0.89 / 0.48 / 0.07 for
  DRC / ResNet / ViT, rotation 1.03 / 0.71-0.76 / 0.73), reached here without any
  edit, search or rollout.
- The null arm reproduces its base to three decimals, which is the control.

## Caveats

- Correlational across arms: length and angle move together in the DRC, so this
  cannot say which one the network's decision uses. The edit searches are what
  separate them.
- The input kernel is ~18% of the DRC axis's length and ~40% of its effect, so
  this describes the input code, not the whole axis.
- One base per architecture for the arm tables; ViT is s9012, the only ViT base
  with an axis.
- BC's arms train for 1k steps against the RL arms' 400k, so less weight
  movement is expected everywhere. What makes the comparison fair is that those
  same arms carry a large, reliable axis (\|axis\| 2.7, reliability 0.96)
  elsewhere in the model: weights moved, just not the input code.

## Follow-up worth running (raised 2026-09-17 in discussion, not yet done)

The rotation is length-preserving, so at a colour-1 object the edited network
sees what the unedited one would see with colour 1 at intensity `1/N` and
colour 0 at `s/N`, where `s = t*|K1|/|K0|` and `N = sqrt(1 + 2t cos(theta) + t^2)`.
Worked for DRC s1234 at t = 0.2: colour 1 dimmed to 0.851, colour 0 painted on
at 0.089.

Those two components push tau in OPPOSITE directions -- dimming the poor colour
raises tau, painting the rich colour on it lowers tau -- so the rotation's
measured effect is painting minus self-dimming, and **understates** the
identity-painting alone. A cleaner probe of identity is the un-normalised edit
`K1 -> K1 + s*K0` (paint without dimming). It is not length-preserving, which is
why it was not the default, but it separates "looks partly like the other
colour" from "looks like less of itself".

An angle measured in FUNCTION space -- the cosine between the two colours'
actual output contributions over real mazes -- would also avoid the flat
Euclidean metric over all 288 weights that the degrees here assume. The
recolouring interpretation is metric-free; the numbers in degrees are not.
