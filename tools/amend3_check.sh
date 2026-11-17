#!/bin/bash
# Amendment 3 checks on the small fixture (code test only).
set -u
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
PY=/home/claude/a15env/bin/python
CODE=../s10_engine/run_s9.py
# (a) regression: without an alias table or EMBER labels, harmonise reproduces amendment 2 on fixture F2
rm -rf G2 && mkdir -p G2/work G2/data && cp -r F2/data/I1 G2/data/ && cp F2/work/*_records*.jsonl.gz G2/work/
$PY $CODE --root G2 harmonise > G2_harmonise.log 2>&1 || { echo "harmonise failed"; tail -20 G2_harmonise.log; }
for f in work/hook_set.json work/split_I1_o1.json work/split_I1_o2.json work/split_I1_o3.json work/family_sets.json work/split_I2_grouped.json; do
  if cmp -s F2/$f G2/$f; then echo "IDENTICAL $f"; else echo "DIFFERENT $f"; fi
done
python3 - <<'PY'
import json
a = json.load(open("F2/outputs/harmonise_counts.json")); b = json.load(open("G2/outputs/harmonise_counts.json"))
same = all(a[k] == {kk: vv for kk, vv in b[k].items() if kk in a[k]} if isinstance(a[k], dict) else a[k] == b[k] for k in a)
print("harmonise_counts: earlier keys equal:", same, "| new keys:", sorted(set(b) - set(a)),
      "| new per-origin keys:", sorted(set(b["origin1"]) - set(a["origin1"])))
print("I2_test_traces_shared by origin:", [b[f"origin{o}"]["I2_test_traces_shared"] for o in (1, 2, 3)],
      "| K4_I1_meld_in_cape:", {k: b["K4_I1_meld_in_cape"][k] for k in ("n_both_named", "agreement")},
      "| K4_I2_avclass_in_cape:", {k: b["K4_I2_avclass_in_cape"][k] for k in ("n_both_named", "agreement")})
PY
