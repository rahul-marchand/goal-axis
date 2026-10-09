"""The array-statistics half of ``goalmisgen.analysis`` imports without JAX.

``analysis/__init__`` used to import ``activations`` eagerly, which reaches
cleanba, so even ``metrics`` dragged a JAX runtime into plotting scripts and
spawned workers. Each probe runs in a fresh interpreter: inside pytest, some
other test has always imported JAX already.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

LIGHT = [
    "arm_length",
    "behaviour",
    "control_writes",
    "far_dose",
    "fields",
    "flips",
    "geometry",
    "layers",
    "metrics",
    "norm_survival",
    "plans",
    "probes",
    "readout_fit",
    "route_targets",
    "separability",
    "spectrum",
    "steering",
    "tables",
    "targets",
    "token_budget",
    "weights",
]


@pytest.mark.parametrize("module", LIGHT)
def test_light_module_is_jax_free(module: str) -> None:
    probe = (
        f"import sys; import goalmisgen.analysis.{module}; "
        "loaded = [m for m in ('jax', 'flax', 'cleanba') if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_package_reexports_resolve_lazily() -> None:
    probe = (
        "import sys; from goalmisgen.analysis import collect_episode_outcomes, summarise, metrics; "
        "assert 'jax' not in sys.modules; "
        "from goalmisgen.analysis import collect_rollouts, Rollout; "
        "import goalmisgen.analysis.activations as a; "
        "assert collect_rollouts is a.collect_rollouts and 'cleanba' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_unknown_name_is_an_attribute_error() -> None:
    import goalmisgen.analysis

    with pytest.raises(AttributeError):
        goalmisgen.analysis.no_such_name  # noqa: B018


def test_every_export_is_listed() -> None:
    import goalmisgen.analysis

    assert sorted(goalmisgen.analysis.__all__) == sorted(goalmisgen.analysis._EXPORTS)


def test_analysis_does_not_import_edits() -> None:
    """The dependency runs one way: the edit search builds on analysis, never the reverse."""
    probe = (
        "import pkgutil, sys, importlib, goalmisgen.analysis as a; "
        "[importlib.import_module(f'goalmisgen.analysis.{m.name}') for m in pkgutil.iter_modules(a.__path__)]; "
        "loaded = [m for m in sys.modules if m.startswith('goalmisgen.edits')]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
