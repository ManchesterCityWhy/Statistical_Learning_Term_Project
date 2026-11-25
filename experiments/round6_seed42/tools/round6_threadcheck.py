"""Round 6: are the anchor lineage calls and the harmonisation shift independent of the BLAS thread count?

Prints SHA-256 of (nearest E10.5 type of every anchor nucleus, shared shift, first 200 harmonised nuclei) for the
export inputs. Run it under different OMP/OPENBLAS thread settings and compare the three hashes.

    THREADS=1 ... python -m experiments.round6_threadcheck
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from experiments.round5_anchor import load_anchor
from experiments.round6_anchor import common_shift, harmonise
from experiments.round6_common import REAL, Mem
from t1_pipeline.data import board_spec


def main():
    _, genes = board_spec("T1:test")
    ax, _ = load_anchor(genes, verbose=False)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    lib3 = np.flatnonzero(e10.lib == "3")
    shift, ctx = common_shift(e10, lib3, ax)
    nuclei = harmonise(ax[:200].toarray().astype(np.float32), shift, 1.0)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    print("threads", os.environ.get("OMP_NUM_THREADS"), "| near", h(np.array(ctx["near"], dtype="U40")), "| shift", h(shift),
          "| harmonised nuclei", h(nuclei), "| lineages", ctx["info"]["lineages"], flush=True)


if __name__ == "__main__":
    main()
