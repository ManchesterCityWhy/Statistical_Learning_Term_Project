import sys, re, inspect
sys.path.insert(0, ".")
import t1_pipeline.evaluation  # sets up the path to the official module
import common.core_metrics as cm
print("file:", cm.__file__)
for name in ("pseudobulk", "de_genes", "_signed_overlap", "_ranks", "de_score", "de_direction", "variogram_score", "mmd_unbiased"):
    f = getattr(cm, name, None)
    print("=====", name)
    print(inspect.getsource(f)[:1500] if f is not None else "NOT FOUND")
