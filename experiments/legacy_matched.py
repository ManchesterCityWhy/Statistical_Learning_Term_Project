from pathlib import Path
import numpy as np,pandas as pd
from t1_pipeline.data import Stage,board_spec
from t1_pipeline.evaluation import metric_panel,calibrated_score
from experiments.bulk_trend import load_trend,apply_trend
from experiments.geo_transport import renormalize
_,g=board_spec('T1:val');a,b,t=[Stage.open(Path(f'data/T1/{s}_RNA.h5ad'),g) for s in ['E8.5','E9.5','E10.5']]
ix=np.load('runs/expanded_seed42/indices.npz');rows=ix['reference'];ref=b.dense(rows);pred=ref.copy();la=a.labels;lb=b.labels
for label in sorted(set(la)&set(lb)):
 mask=lb[rows]==label;delta=b.mean(np.flatnonzero(lb==label))-a.mean(np.flatnonzero(la==label));pred[mask]=np.maximum(pred[mask]+delta*(pred[mask]>0),0)
trend,_=load_trend(g);pred=apply_trend(renormalize(pred),trend,.75/5,False,42);out=[]
for split,ceilkey in [('development','ceiling'),('confirmation','confirmation_ceiling')]:
 target=t.dense(ix[split]);ceil=t.dense(ix[ceilkey]);f=metric_panel(ref,target,ref);c=metric_panel(ceil,target,ref);m=metric_panel(pred,target,ref);out.append({'split':split,'candidate':'legacy_bulk_type_075',**m,'score':calibrated_score(m,f,c)['local_score']})
pd.DataFrame(out).to_csv('runs/expanded_seed42/legacy_matched.csv',index=False)
for s in [a,b,t]:s.close()
