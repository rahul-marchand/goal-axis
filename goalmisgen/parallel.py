"""Worker pools that do not inherit a forked JAX runtime.

Level and demonstration generation are both CPU-bound, embarrassingly parallel,
and JAX-free: ``envs.dataset`` and ``offline.demos`` import numpy and nothing
else heavy. Their *entry points* are not. ``scripts/generate_levels.py`` and
``scripts/generate_demos.py`` both import ``configs.env``, which reaches cleanba
and therefore JAX, so by the time either reaches its pool the parent process has
a multithreaded JAX runtime in it.

Forking that is the documented deadlock -- ``os.fork()`` copies one thread and
leaves every lock the others held closed forever. Python warns about it, and
``configs/presets.py`` already records the same hazard on the evaluation path.
It has not bitten yet, which is the worst way for a hazard like this to sit:
when it does, generation *hangs* rather than failing, so it reads as a slow job
while still billing for the machine, exactly as the evaluation stall did.

``spawn`` starts each worker from a fresh interpreter, so there is no inherited
lock state to deadlock on. The cost is that the child re-imports the parent's
``__main__``, and both entry points guard theirs with ``if __name__ ==
"__main__"``, so what it re-imports is the module body and not another run.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import Pool
from pathlib import Path


def worker_pool(workers: int) -> Pool:
    """A pool of ``workers`` processes, started clean rather than forked.

    Used as a context manager, like ``mp.Pool``. Everything passed to it has to
    pickle, which was already true under fork-with-arguments.
    """
    return mp.get_context("spawn").Pool(workers)


def usable_cpus(root: Path = Path("/")) -> int:
    """CPUs we are actually allowed to use.

    ``multiprocessing.cpu_count()`` reports the host's cores, which inside a
    container is wildly optimistic — a cloud pod may show 48 while its cgroup
    quota permits 5. Oversubscribing that badly makes generation slower, not
    faster, so read the quota where one exists.

    Both cgroup layouts have to be tried. Reading only ``cpu.max`` covers v2 and
    silently falls through to the host's core count on a v1 host, which is how a
    pod with a 5.1-core quota came to be generating levels across 48 workers,
    each getting a tenth of a core. ``nproc`` and ``cpuset`` both said 48 there;
    only ``cpu.cfs_quota_us`` said otherwise.
    """
    cgroup = root / "sys/fs/cgroup"
    try:  # cgroup v2
        quota, period = (cgroup / "cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
        return mp.cpu_count()
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = float((cgroup / "cpu/cpu.cfs_quota_us").read_text())
        period = float((cgroup / "cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return mp.cpu_count()
