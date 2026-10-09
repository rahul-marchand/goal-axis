"""An ordinary ``013`` value arm with some modules frozen.

    uv run python scripts/frozen_arm.py --freeze "input conv" "conv 1" -- \
        CHECKPOINT --objective-values 1 0.2 --levels ... --run-dir .../arms-frozen/o1-030@400k \
        --steps 400000 --lr 0.0001 --checkpoints 4 --size 11 --seed 1237

Everything after ``--`` is passed to ``experiments/013_value_axis.py`` unchanged,
so a frozen arm can be launched with exactly the argv of an existing arm (same
seed, same levels, same rate) and differ from it only in what SGD may touch.
The freezing is done from our side (:mod:`goalmisgen.edits.freeze`); cleanba is
not edited. ``FROZEN.json`` beside the checkpoints records which modules were held.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    if "--" not in sys.argv:
        sys.exit("usage: frozen_arm.py --freeze MODULE [MODULE ...] -- <013 arguments>")
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", nargs="+", required=True)
    args = parser.parse_args(sys.argv[1:split])
    rest = sys.argv[split + 1 :]

    from goalmisgen.edits.freeze import frozen_modules

    spec = importlib.util.spec_from_file_location("value_axis_013", ROOT / "experiments" / "013_value_axis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = Path(rest[rest.index("--run-dir") + 1])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "FROZEN.json").write_text(json.dumps({"frozen": args.freeze, "argv": rest}, indent=1))
    print(f"frozen modules  {args.freeze}")

    sys.argv = [str(ROOT / "experiments" / "013_value_axis.py"), *rest]
    with frozen_modules(args.freeze):
        module.main()


if __name__ == "__main__":
    main()
