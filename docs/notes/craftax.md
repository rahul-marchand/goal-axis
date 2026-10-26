# Craftax stream: session notes (2026-09-15..17)

Copied into the repo on 2026-10-26 from working memory, trimmed to decisions,
findings and the open stage-4 questions (THREADS K.1-K.5). The design and
status are `docs/Craftax.md`; tables `results/craftax/`; figures
`figures/craftax/`.

## Decisions

- Craftax is an engine, not a task: `craftax==1.6.1` pinned and unmodified,
  everything of ours in `goalmisgen/craftax/`, and only `engine.py` imports JAX.
  Engine rollout is pinned to the CPU: on a GPU it is launch-bound (400 s per
  1,024 levels).
- Canonical expert: greedy, nearest-first, row-major ties, up/down/left/right
  move preference (`craftax/routes.py`), so every expert decision is inferable
  from the map. The first stage-4 expert ranked plans over permutations, which
  the map does not determine; its base reached 3% with 85% token accuracy on
  train and held-out alike, first error in the first leg on 60% of fields.
- Stage 4 is receding-horizon: facing and inventory planes, training on every
  state, one action decoded per pass through the engine.
- bf16 fails on the RTX 5090 with jax 0.4.34 (an LLVM bf16→f16 error); runs are
  fp32.

## Findings beyond `docs/Craftax.md`

- The blind-table stage-4 base (`cxcraft15.s1`) failed because the crafting
  observation had no channel for a placed table (rendered as grass), so the
  state was not Markov: the coal chain worked, the iron chain (which returns to
  the table) could not. DAgger fitted the visited states without closed-loop
  gain, which is what exposed it. Adding `TABLE_CHANNEL` fixed it: the
  warm-started `cxcraft15.s1t` reaches 98% after 2.5k steps. Even the
  blind-table base had an axis (slopes ±18-21 against the expert's 20,
  reliability 0.6, cos −0.86).
- Price probe on the blind-table base (`craft_price_probe.py`): handed a stone
  pickaxe it takes iron 100% of the time (83% reach); handed the ingredients it
  switches to iron 100% but completes 8-35%. A switch on what is in hand, not a
  graded price.

## Open: is the stage-4 axis an input gain? (THREADS K.3)

Behaviour is the same under a gain on the iron cue, so reproducing the axis does
not force H2 (`docs/notes/gain-vs-factorised.md`). The tests that decide it:

- where the axis's norm sits: the embedding rows for the iron cue against
  downstream weights;
- a brightness control on the iron plane;
- restricted writes, as for the maze agents;
- propagation: does raised iron value change the instrumental steps before the
  iron is mined? The price probe on the o0 ±0.45 arms was run for this
  (`results/craftax/price.cxcraft15.s1t.o0{+,-}045.txt`) and is unanalysed.
