#!/usr/bin/env python3
"""Build the basin figure fragment: inject data.json into template.html.

Regenerate after new experiments by overwriting data.json (same schema:
agent, sweep, baseValues, stepPenalty, alphas, pop, levels) and re-running.
Output: ../../fragments/basin.html — splice into the post with <!-- fig: basin -->.
"""
import json
import pathlib

src = pathlib.Path(__file__).resolve().parent
data = json.dumps(json.loads((src / "data.json").read_text()), separators=(",", ":"))
fragment = (src / "template.html").read_text().replace("__DATA__", data)
out = src.parent.parent / "fragments" / "basin.html"
out.parent.mkdir(exist_ok=True)
out.write_text(fragment)
print(f"wrote {out} ({len(fragment)//1024} KB)")
