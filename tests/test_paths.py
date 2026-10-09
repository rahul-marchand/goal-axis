"""Where the data volume is, and the names joined onto it."""

from pathlib import Path

from goalmisgen import paths


def test_the_root_defaults_to_where_the_pods_mount_the_volume(monkeypatch) -> None:
    monkeypatch.delenv(paths.DATA_ENV, raising=False)
    assert paths.data_root() == Path("/workspace/data")


def test_the_root_follows_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.run_dir("novalue11g1.s1234") == tmp_path / "runs" / "novalue11g1.s1234"
    assert paths.offline_demos("train") == tmp_path / "offline" / "demos" / "train.rho100"


def test_the_paths_scripts_used_to_spell_out(monkeypatch) -> None:
    monkeypatch.delenv(paths.DATA_ENV, raising=False)
    assert str(paths.level_dataset()) == "/workspace/data/levels/values/1.00-0.50@500k"
    assert str(paths.level_dataset(n_levels=1_000_000)) == "/workspace/data/levels/values/1.00-0.50@1M"
    assert str(paths.offline_demos()) == "/workspace/data/offline/demos/test.rho100"
    assert str(paths.offline_runs_dir()) == "/workspace/data/offline/runs"
    assert paths.figures_data_dir() / "drc" / "basin" == Path("/workspace/data/figures-data/drc/basin")


def test_the_repo_root_holds_this_package() -> None:
    assert (paths.repo_root() / "goalmisgen" / "paths.py").is_file()
