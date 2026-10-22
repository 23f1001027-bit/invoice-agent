"""Try decisions locally without spending a grader check.

1. In Render logs, copy the JSON after "SAMPLE PACKAGE:" into sample.json
   (or save a whole batch's "packages" list there).
2. AIPIPE_TOKEN=... python local_test.py sample.json
"""
import json, os, sys
os.environ.setdefault("DB_PATH", ":memory:")
import core, llm

data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "sample.json"))
pkgs = data if isinstance(data, list) else data.get("packages", [data])
pm = {}
print(llm.render(pkgs[0], pm)[:3000], "\n...")
for p, d in zip(pkgs, core.decide(pkgs, llm.llm)):
    print(json.dumps({"pkg": p.get("packageId"), **d}, indent=1, ensure_ascii=False))
