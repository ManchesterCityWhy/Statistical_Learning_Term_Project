"""Compare two score tables written by round6_final (same keys): max absolute difference per column."""
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
key = [c for c in ("config", "scenario", "proxy") if c in a.columns]
a = a[a.get("seed", 42) == 42] if "seed" in a else a
b = b[b.get("seed", 42) == 42] if "seed" in b else b
m = a.merge(b, on=key, suffixes=("_old", "_new"))
cols = ["score", "de_score", "de_direction", "mmd_u", "variogram"]
print(sys.argv[1].split("/")[-1], "rows old", len(a), "new", len(b), "matched", len(m),
      {c: float((m[c + "_old"] - m[c + "_new"]).abs().max()) for c in cols})
