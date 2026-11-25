"""Round 5: describe the exported candidates (manifest fields, donor composition, depth of donors)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_common import CARDIAC, REAL, SEED, Mem
from experiments.round5_exp2 import deconv_q
from experiments.round5_exp4 import depth_of
from t1_pipeline.data import board_spec


def main():
    pd.set_option("display.width", 250)
    for d in sorted((REAL / "predictions").glob("round5_*")):
        m = json.loads((d / "manifest.json").read_text())
        print(d.name, {k: m[k] for k in ("pool_cells", "anchor_cells", "nonzero_fraction", "nonzero_fraction_base",
                                         "library_min", "library_max", "regression_check")}, m["validation"])
    _, genes = board_spec("T1:test")
    trend, _ = load_trend(genes)
    source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    depth = depth_of(source.X)
    table = {}
    for pool_name, h in (("all", 1), ("all", 2), ("cardiac", 2), ("lib3", 2)):
        pool = {"all": np.arange(source.n_cells), "lib3": np.flatnonzero(source.lib == "3"),
                "cardiac": np.flatnonzero(np.isin(source.labels, list(CARDIAC)))}[pool_name]
        labels = source.labels[pool]
        names = np.unique(labels)
        prof = np.stack([np.asarray(source.X[pool[labels == g]].expm1().mean(axis=0)).ravel() for g in names], axis=1)
        p = np.array([(labels == g).mean() for g in names])
        q = deconv_q(prof, p, trend, 3.0 * h, 10)
        w = np.zeros(len(pool))
        for g, prob in zip(names, q):
            w[labels == g] = prob / (labels == g).sum()
        rows = pool[np.random.default_rng(SEED).choice(len(pool), 2000, replace=False, p=w / w.sum())]
        got = pd.Series(source.labels[rows]).value_counts(normalize=True)
        table[f"{pool_name}_h{h}"] = got
        print(pool_name, h, "donor library shares", pd.Series(source.lib[rows]).value_counts(normalize=True).round(3).to_dict(),
              "median donor UMI", float(np.median(depth[rows])))
    print((100 * pd.DataFrame(table).fillna(0)).round(1).sort_values("cardiac_h2", ascending=False).to_string())


if __name__ == "__main__":
    main()
