"""What things on the data volume are called, and how to read a name back.

A directory name is the only piece of metadata that is visible before anything
is loaded, so it is the only one that can stop a mistake rather than record it.
The previous layout recorded two facts in prose that it should have carried in
the path, and both of them cost something:

* **Which agent an arm belongs to.** ``valueaxis/runs/v070`` was seed 1234 and
  ``valueaxis_s5678/runs/v070`` was seed 5678, so one seed was named and the
  other was a convention. A third seed has nowhere to go that is not a third
  convention.
* **How long an arm trained for.** Seed 1234's arms ran 3M steps and seed
  5678's ran 750k, and every quantity that scales with fine-tune length is
  therefore incomparable between them. ``results/seed-comparison.txt`` says so
  in a paragraph, which works exactly until somebody does not read it.

So an arm is ``<sweep><offset>@<steps>`` beneath the agent it was fine-tuned
from, and the two things that must never be silently mixed are both in the
name. ``014`` already refuses to fit arms at different budgets in one directory
(``arm_checkpoints``); putting the budget in the path extends that refusal
across directories, where the comparison is not checked at all.

The scheme is otherwise deliberately flat. Arms from a seven-point grid and a
thirty-point grid at the same length belong in the same directory and *should*
be fitted together — more leverage on the same axis — so design generation is
not part of the name. Only length is, because only length makes arms
incompatible.

One wart worth knowing: ``+`` sorts before ``-`` in ASCII, so a directory
listing puts every negative offset last. Nothing depends on it — ``014`` keys
arms by their parsed value — but anything presenting arms in offset order has to
parse the name rather than sort the listing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ``o1+020@750k``: the colour-1 sweep, trained at 0.20 above the base value, for
# 750,000 steps. ``o1+020@750k.r2`` is the same fine-tune run again from a
# different seed -- a *replicate*. The original grid carries no suffix, so every
# directory that existed before replicates did still parses, and a replicate can
# never silently collide with the arm it re-runs. Composition arms that move
# several values at once do not have a single offset and are named separately;
# see ``is_composition_arm``.
_ARM = re.compile(
    r"^(?P<sweep>[a-z][a-z0-9]*)(?P<sign>[+-])(?P<offset>\d{3})@(?P<steps>\d+[kM])(?:\.r(?P<replicate>[2-9]\d*))?$"
)


# Which sweep families exist, and whether their arms enter the fitted axis.
#
# ``o`` is the symmetric grid the axis is fitted from. ``x`` is a one-sided
# extension -- arms past the point where the swept objective overtakes the other,
# which a mirrored grid cannot reach without asking for a negative reward. ``m``
# moves several values at once. ``k`` is ``x`` with the colours pinned to
# objectives rather than to value rank, which is the only way an arm past parity
# asks for a reversed preference rather than a relabelled one. ``x``, ``k`` and
# ``m`` are *held out of every fit*: all three are unbalanced about the base, and an unbalanced grid lets the common
# fine-tuning component leak into the axis, which is the failure the symmetric
# design exists to prevent. That is a statistical claim, not a filing preference,
# which is why it lives here rather than in whichever script last needed it.
FITTED_FAMILIES: frozenset[str] = frozenset({"o"})
HELD_OUT_FAMILIES: frozenset[str] = frozenset({"x", "m", "k"})

# Families whose levels pin colour i to objective i rather than to value rank.
# A property of the family, not a per-run flag, so an arm's directory name says
# how its levels were coloured and the two can never disagree.
COLOUR_KEYED_FAMILIES: frozenset[str] = frozenset({"k"})


@dataclass(frozen=True)
class ArmName:
    """One fine-tuning arm, as its directory name says it is."""

    sweep: str
    offset: float
    steps: int
    replicate: int = 1
    """Which independent draw of this fine-tune the directory holds.

    Replicate 1 is the original grid and carries no suffix -- the naming rule,
    not a default that happens to hold. Seed-band figures need the same arm
    trained more than once, and two runs of ``o1+020@400k`` must not overwrite
    each other, which is exactly the class of mistake this module exists to
    make unavailable.
    """

    @property
    def is_null(self) -> bool:
        """Fine-tuned onto the value it already had, so it measures drift."""
        return self.offset == 0.0


def _round_tag(n: int, what: str) -> str:
    """``750000`` -> ``750k``, ``3000000`` -> ``3M``.

    Only exact multiples get the short form, so a tag can always be read back to
    the number that produced it. An arm at 1,234,567 steps would round to the
    same tag as one at 1.2M and silently claim to be comparable with it, which
    is the failure this whole module exists to prevent.
    """
    if n <= 0:
        raise ValueError(f"{what} cannot be {n}")
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    raise ValueError(
        f"{n:,} has no exact tag, so it cannot name {what}; these should be round numbers "
        "so that two things claiming the same size really have it"
    )


def _parse_round_tag(tag: str, what: str) -> int:
    match = re.fullmatch(r"(\d+)([kM])", tag)
    if match is None:
        raise ValueError(f"{tag!r} is not {what}, which looks like '750k' or '3M'")
    return int(match.group(1)) * (1_000 if match.group(2) == "k" else 1_000_000)


def steps_tag(steps: int) -> str:
    """How long an arm trained for, as it appears in the arm's directory name."""
    return _round_tag(steps, "a number of training steps")


def parse_steps_tag(tag: str) -> int:
    """``750k`` -> ``750000``. The inverse of :func:`steps_tag`."""
    return _parse_round_tag(tag, "a step tag")


def offset_tag(offset: float) -> str:
    """``+0.2`` -> ``+020``, ``-0.05`` -> ``-005``, ``0.0`` -> ``+000``.

    Signed always, so the null arm sorts with the rest of its sweep rather than
    reading as a different kind of thing, and so a missing sign is a malformed
    name rather than an assumed positive.
    """
    hundredths = round(offset * 100)
    if abs(hundredths / 100 - offset) > 1e-9:
        raise ValueError(f"offset {offset} is finer than the hundredth of a value the name can carry")
    return f"{'-' if hundredths < 0 else '+'}{abs(hundredths):03d}"


def arm_dirname(sweep: str, offset: float, steps: int, replicate: int = 1) -> str:
    """The directory one arm lives in, e.g. ``o1+020@750k`` (``.r2`` for a replicate)."""
    if not re.fullmatch(r"[a-z][a-z0-9]*", sweep):
        raise ValueError(f"sweep {sweep!r} should be a short lower-case tag like 'o0', 'o1' or 'o2'")
    if replicate < 1:
        raise ValueError(f"replicate counts from 1, got {replicate}")
    suffix = "" if replicate == 1 else f".r{replicate}"
    return f"{sweep}{offset_tag(offset)}@{steps_tag(steps)}{suffix}"


def parse_arm_dirname(name: str) -> ArmName | None:
    """Read an arm's name back, or ``None`` if it is not one.

    Returning ``None`` rather than raising is deliberate: callers walk a
    directory that also holds composition arms and whatever else has been put
    there, and the useful behaviour is to report what was skipped rather than to
    stop on the first thing that is not an arm.
    """
    match = _ARM.fullmatch(name)
    if match is None:
        return None
    offset = int(match.group("offset")) / 100
    return ArmName(
        sweep=match.group("sweep"),
        offset=-offset if match.group("sign") == "-" else offset,
        steps=parse_steps_tag(match.group("steps")),
        replicate=int(match.group("replicate") or 1),
    )


_CHECKPOINT = re.compile(r"^cp_(?P<steps>\d+)$")


def parse_checkpoint_dirname(name: str) -> int | None:
    """``cp_070103040`` -> ``70103040``, or ``None`` if it is not a checkpoint.

    The zero padding is *not* fixed across runs. cleanba's writer pads to
    ``ceil(log10(total_timesteps))`` digits, so the same 70,103,040 steps is
    ``cp_070103040`` in the 150M ``novalue11`` runs and ``cp_70103040`` in the
    80M ``threeobj`` ones. Anything comparing checkpoints across runs — or
    building a name to look one up — has to read the number back rather than
    match the string, which is what this exists for.
    """
    match = _CHECKPOINT.fullmatch(name)
    return None if match is None else int(match.group("steps"))


def rung_agent_name(agent: str, checkpoint: str) -> str:
    """``novalue11.s1234`` at ``cp_070103040`` -> ``novalue11.s1234.at70103040``.

    A *rung* is the same agent seen earlier in its own training: the value sweep
    run again from an intermediate checkpoint, so the axis fitted there can be
    compared against the one fitted at the end.

    Named for the step count rather than for the checkpoint directory, so that
    the two paddings :func:`parse_checkpoint_dirname` describes give one name for
    one point in training. The earlier shell version stripped a literal ``cp_0``
    prefix instead, which silently produced ``...atcp_100147200`` for every
    checkpoint past 100M — the ones with no leading zero to strip.
    """
    steps = parse_checkpoint_dirname(checkpoint)
    if steps is None:
        raise ValueError(f"{checkpoint!r} is not a checkpoint directory name, which looks like 'cp_070103040'")
    return f"{agent}.at{steps}"


def arm_is_complete(run_dir: Path, steps: int) -> bool:
    """Did this arm reach the length its name claims?

    Presence of *a* checkpoint is not enough. An arm killed part way leaves the
    ones it had already saved, and resuming would skip it and then fit it as
    though it had run the full budget -- a 200k arm sitting in the grid under a
    @400k name, which is precisely the silent incomparability the naming scheme
    exists to prevent. Checkpoints are named in steps, so the last one says how
    far the arm actually got.
    """
    saved = [parse_checkpoint_dirname(p.name) for p in run_dir.glob("local-files/cp_*")]
    reached = [steps_reached for steps_reached in saved if steps_reached is not None]
    return bool(reached) and max(reached) >= 0.98 * steps


def is_composition_arm(name: str) -> bool:
    """``m_120_045_030@1M`` — several values moved at once.

    These are held out of every fit by construction, so they are named for the
    values they carry rather than for an offset they do not have.
    """
    return re.fullmatch(r"m(_\d{3})+@\d+[kM]", name) is not None


def composition_arm_dirname(values: Iterable[float], steps: int) -> str:
    """``m_120_045_030@1M`` from the values it was trained on."""
    tags = "".join(f"_{round(v * 100):03d}" for v in values)
    if not tags:
        raise ValueError("a composition arm needs at least one value")
    return f"m{tags}@{steps_tag(steps)}"


def composition_arm_values(name: str) -> tuple[float, ...] | None:
    """The values a composition arm was trained at, or ``None`` if it is not one.

    The inverse of :func:`composition_arm_dirname`. Composition arms carry their
    values outright rather than an offset, because they move several at once and
    so have no single offset to be named for.
    """
    if not is_composition_arm(name):
        return None
    return tuple(int(part) / 100 for part in name.split("@")[0].split("_")[1:])


def arm_trained_values(name: str, base_values: tuple[float, ...]) -> tuple[float, ...] | None:
    """What an arm's objectives paid, from its directory name alone.

    Reading them from the name keeps the analysis honest if a grid is edited: a
    table in a script can drift out of step with what was actually run, a
    directory name cannot.
    """
    parsed = parse_arm_dirname(name)
    if parsed is not None:
        index = int(parsed.sweep[1:])
        if not 0 <= index < len(base_values):
            return None
        values = list(base_values)
        values[index] = round(values[index] + parsed.offset, 10)
        return tuple(values)
    return composition_arm_values(name)


def values_tag(values: Iterable[float]) -> str:
    """``(1.0, 0.5)`` -> ``1.00-0.50``, the key of a shared level dataset.

    Datasets are keyed by what the objectives pay because that is the only thing
    that distinguishes them: ``FixedValues`` consumes no randomness, so two
    datasets generated at one seed with different values have byte-identical
    layouts. Keying by content rather than by the campaign that first needed it
    is what lets one dataset serve every seed.
    """
    values = list(values)
    if not values:
        raise ValueError("a level dataset needs at least one objective value")
    if any(v < 0 for v in values):
        raise ValueError(f"objective values are rewards and cannot be negative: {values}")
    return "-".join(f"{v:.2f}" for v in values)


def parse_values_tag(tag: str) -> tuple[float, ...]:
    """``1.00-0.50`` -> ``(1.0, 0.5)``. The inverse of :func:`values_tag`."""
    if not re.fullmatch(r"\d+\.\d{2}(-\d+\.\d{2})*", tag):
        raise ValueError(f"{tag!r} is not a values tag like '1.00-0.50'")
    return tuple(float(part) for part in tag.split("-"))


def dataset_dirname(values: Iterable[float], n_levels: int) -> str:
    """``1.00-0.50@1M`` — the key one shared level dataset lives under.

    The level count is part of the key and not decoration. Two datasets at the
    same values but different counts are *not* interchangeable: the layouts are
    a deterministic function of the seed and the number of levels asked for, so
    a 500k dataset is not the first half of a 1M one. Keying on values alone
    would quietly merge ``levels11`` with ``valueaxis/levels/v050`` — same
    values, 1M levels against 500k — and every run downstream would train on
    mazes it did not train on.

    What *is* shared is everything at one key: ``FixedValues`` consumes no
    randomness, so datasets differing only in what the objectives pay have
    byte-identical layouts, and one copy serves every seed and every campaign.
    """
    return f"{values_tag(values)}@{_round_tag(n_levels, 'a number of levels')}"


def iter_arms(arms: Path, sweep: str, steps: int | None = None, replicate: int = 1) -> Iterator[tuple[ArmName, Path]]:
    """Every arm directory of one sweep, in listing order, with its parsed name.

    The one place that decides which directories belong to a sweep, so every
    consumer agrees on it. ``sweep`` is the full tag (``o1``, ``k0``), which is
    what keeps held-out families out of anything asking for the fitted one.
    ``replicate`` defaults to the original grid: replicates are separate draws of
    the same arms, and a consumer keying arms by offset would otherwise let a
    ``.r2`` directory silently replace the arm it re-runs. ``steps=None`` accepts
    every length -- callers that fit must pass it, see :func:`discover_arms`.
    """
    if not arms.is_dir():
        return
    for run in sorted(arms.iterdir()):
        parsed = parse_arm_dirname(run.name)
        if parsed is None or parsed.sweep != sweep or parsed.replicate != replicate:
            continue
        if steps is not None and parsed.steps != steps:
            continue
        yield parsed, run


def discover_arms(
    arms: Path,
    objective: int,
    base_value: float,
    steps: int | None = None,
    at: int = -1,
    family: str = "o",
    replicate: int = 1,
) -> dict[float, Path]:
    """One sweep's arms, keyed by the value each was trained at.

    The key is the value rather than the offset because that is what the analysis
    has always regressed against, so the scripts that consume this did not have
    to change shape when the directories were renamed.

    ``steps`` is not optional in spirit. An agent's ``arms/`` directory now holds
    every sweep ever run against it, and arms of different lengths are not
    comparable -- that is the whole reason the length is in the name. So a
    directory holding more than one length raises unless the caller says which it
    means. ``014`` already warned about this within a directory; this makes the
    mistake unavailable rather than reported.
    """
    found: dict[float, Path] = {}
    lengths: set[int] = set()
    # ``family`` is what keeps the held-out arms out of a fit: the default sees
    # only ``o``, so ``x``, ``k`` and ``m`` arms are invisible unless asked for by
    # name. See FITTED_FAMILIES.
    for parsed, run in iter_arms(arms, f"{family}{objective}", steps=steps, replicate=replicate):
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            continue
        try:
            found[round(base_value + parsed.offset, 10)] = checkpoints[at]
        except IndexError:
            print(f"  {run.name}: no checkpoint at index {at}, skipping")
            continue
        lengths.add(parsed.steps)
    if steps is None and len(lengths) > 1:
        raise ValueError(
            f"{arms} holds arms at {sorted(lengths)} steps. Fitting them together would mix "
            "budgets, which is what putting the length in the name exists to prevent -- pass "
            "steps= to say which sweep you mean."
        )
    return found


def arm_lengths(arms: Path) -> set[int]:
    """Every arm budget present in one agent's ``arms/`` directory."""
    if not arms.is_dir():
        return set()
    return {p.steps for run in arms.iterdir() if (p := parse_arm_dirname(run.name)) is not None}


def sweep_family(prefix: str) -> str:
    """Which family a sweep belongs to: ``o`` fitted; ``x``, ``k`` and ``m`` held out."""
    cleaned = prefix.strip().rstrip("_")
    if cleaned in ("v", "c"):
        return "o"
    if cleaned and cleaned[0].isalpha() and cleaned[0] in FITTED_FAMILIES | HELD_OUT_FAMILIES:
        return cleaned[0]
    return "o"


def is_colour_keyed(sweep: str) -> bool:
    """Whether an arm of this sweep trains on colour-keyed levels; see COLOUR_KEYED_FAMILIES."""
    return sweep_family(sweep) in COLOUR_KEYED_FAMILIES


def sweep_index(prefix: str) -> int:
    """Which objective a sweep moves, from however it is named on the command line.

    ``v`` and ``c`` are the old two-objective names -- ``v`` swept colour 1 and
    ``c`` swept colour 0 -- and are still accepted so that recorded invocations in
    ``results/`` keep working. ``o0``, ``o1``, ``o2`` are what the directories say
    now, and generalise past two objectives.
    """
    cleaned = prefix.strip().rstrip("_")
    if cleaned in ("v", "c"):
        return 1 if cleaned == "v" else 0
    match = re.fullmatch(r"[a-z]?(\d)", cleaned)
    if match is None:
        raise ValueError(f"{prefix!r} does not name a sweep; expected 'o0', 'o1', 'x1', ... (or legacy 'v'/'c')")
    return int(match.group(1))


@dataclass(frozen=True)
class BaseMarker:
    """What ``runs/<agent>/BASE.json`` says: which checkpoint is the agent, and what its objectives pay.

    Everything downstream resolves an agent through this file rather than by
    globbing its checkpoints -- see :mod:`goalmisgen.base_ladder` for the rung
    whose ``local-files`` holds its parent's every checkpoint, where globbing
    silently describes the wrong agent.
    """

    run_dir: Path
    checkpoint: Path
    values: tuple[float, ...]
    payload: dict = field(compare=False, repr=False)
    """The whole file, for the fields only some markers carry (``steps``, ``rung_of``, ``observation``)."""


def read_base_marker(run_dir: Path) -> BaseMarker:
    """Read ``run_dir/BASE.json``. Raises if it is missing; the checkpoint is not checked for existence."""
    run_dir = Path(run_dir)
    marker = run_dir / "BASE.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"{marker} does not exist. It is written by scripts/migrate_volume.py --retire (or a base chain), "
            "and records which checkpoint of this agent everything downstream resolves."
        )
    payload = json.loads(marker.read_text())
    return BaseMarker(
        run_dir=run_dir,
        checkpoint=run_dir / payload["checkpoint"],
        values=tuple(payload["values"]),
        payload=payload,
    )
