"""Which cell types of the file get their mixture partner from the same TYPE and which from the same LINEAGE (fewer than
50 anchor nuclei assigned to the type)?  Counts only."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of
from experiments.round7_method import Anchor7, MIN_CELLS
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:test")
ax, _ = load_anchor(list(genes), verbose=False)
anchor = Anchor7(ax)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", list(genes))
lib3 = np.flatnonzero(np.asarray(e10.lib).astype(str) == "3")
ctx = anchor.context(e10, lib3, "min", "type")
near, lin_a = ctx["near"], ctx["lineage"]
prov = np.load(REAL / "predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3/provenance.npz")
rows, partner = prov["real_rows"], prov["mixture_partner"]
lab = e10.labels[rows]
out = []
for t in np.unique(lab):
    lin = lineage_of(np.array([t]))[0]
    n_type, n_lin = int((near == t).sum()), int((lin_a == lin).sum())
    cells = lab == t
    mixed = int((partner[cells] >= 0).sum())
    same_type = int((near[partner[cells & (partner >= 0)]] == t).sum()) if mixed else 0
    out.append({"type": t, "lineage": lin, "file cells": int(cells.sum()), "mixed": mixed, "anchor nuclei of the type": n_type, "anchor nuclei of the lineage": n_lin,
                "partners of the same type": same_type, "partner pool": "none" if mixed == 0 else ("type" if n_type >= MIN_CELLS else "lineage")})
f = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(f.to_string(index=False))
print("anchor nuclei in total", len(near), "| assigned to Blood", int((lin_a == "Blood").sum()), "| to Other", int((lin_a == "Other").sum()))
print("mixed cells with a partner of the same type:", int(f["partners of the same type"].sum()), "of", int(f["mixed"].sum()))
print("library-3 macrophages:", int((e10.labels[lib3] == "Macrophage").sum()))
