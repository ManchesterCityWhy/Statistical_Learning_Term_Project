import json, sys
from pathlib import Path
m = json.load(open(Path.home() / "statlearn_real/predictions" / sys.argv[1] / "manifest.json"))
print(json.dumps(m.get("selfcheck_same_code_path"), indent=1))
print("base", json.dumps(m["base"]["lineage_shares"]), m["base"]["median_umi_of_source_rows"])
print("anchor", json.dumps(m["anchor"]["lineage_shares"]), m["anchor"]["median_umi_raw_nuclei"], m["anchor"].get("median_umi_resampled"))
print("anchor types", json.dumps(m["anchor"]["nearest_E10.5_celltype_shares"]))
for k, v in m.get("two_witness_shift", {}).items():
    print(k, v["genes_moved"], v["genes_abs_over_0.5"]); print("  up", v["up_top15"]); print("  down", v["down_top15"])
print(m["harmonisation"]["lineages"], m["harmonisation"]["genes_with_shared_shift"])
