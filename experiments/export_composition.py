from pathlib import Path
import json
import numpy as np
from scipy.optimize import nnls
from t1_pipeline.data import Stage,board_spec,write_json,file_hash,validate_prediction
from t1_pipeline.cli import make_prediction_file
from experiments.bulk_trend import load_trend,apply_trend
import pandas as pd
rng=np.random.default_rng(42)
audit=json.loads(Path('runs/final_audit_seed42.json').read_text());cfg=audit['configuration']
_,genes=board_spec('T1:test');b=Stage.open(Path('data/T1/E10.5_RNA.h5ad'),genes);trend,ext=load_trend(genes);labels=b.labels;types=np.unique(labels);means=[];p=[]
for lab in types:
 ids=np.flatnonzero(labels==lab);p.append(len(ids)/b.n_cells);total=np.zeros(len(genes))
 for chunk in np.array_split(ids,max(1,len(ids)//128)):total+=np.expm1(b.dense(chunk)).sum(axis=0,dtype=np.float64)
 means.append(total/len(ids))
profiles=np.stack(means,axis=1);p=np.array(p);base=profiles@p;keep=(base>.1)&(np.std(profiles,axis=1)>.1);design=profiles[keep]/np.sqrt(base[keep,None]+1)
for h in [1,2]:
 desired=base*np.exp(np.clip(trend*cfg['strength']*h,-2,2));desired*=base.sum()/desired.sum();q=nnls(np.vstack([design,10*np.eye(len(p))]),np.r_[desired[keep]/np.sqrt(base[keep]+1),10*p])[0];q/=q.sum()
 weights=np.zeros(b.n_cells)
 for lab,prob in zip(types,q):weights[labels==lab]=prob/(labels==lab).sum()
 rows=rng.choice(b.n_cells,2000,replace=False,p=weights/weights.sum());pred=apply_trend(b.dense(rows),trend,cfg['bulk']*h/5,False,42)
 out=Path('predictions')/f'composition_seed42_h{h}';out.mkdir(parents=True,exist_ok=False);path=out/'E12.5_prediction.h5ad';make_prediction_file(path,pred,genes);v=validate_prediction(path,"T1:test")
 sums=np.expm1(pred.astype(np.float64)).sum(axis=1)
 write_json(out/'manifest.json',{'candidate':cfg,'extrapolation_multiplier':h,'seed':42,'input_E10_5_sha256':file_hash(b.path),'external':ext,'validation':v,'normalization_sum_min':float(sums.min()),'normalization_sum_max':float(sums.max()),'sha256':file_hash(path),'proportions':dict(zip(types,q)),'status':'not_submitted','note':'h1 retains validated correction strength; h2 doubles trend for two-day horizon; neither has E12.5 performance feedback'})
 print(path,flush=True)
b.close()
