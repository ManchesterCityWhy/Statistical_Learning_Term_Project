from pathlib import Path
import numpy as np,pandas as pd
from t1_pipeline.data import Stage,board_spec,write_json
from t1_pipeline.evaluation import metric_panel,calibrated_score
from experiments.bulk_trend import load_trend,apply_trend
rng=np.random.default_rng(42)
folders=['composition_seed42','composition_extended_seed42','composition_refined_seed42']
frames=[]
for folder in folders:
 df=pd.read_csv(Path('runs')/folder/'development.csv');df['run']=folder;frames.append(df)
best=pd.concat(frames).sort_values('score',ascending=False).iloc[0]
import json
cfg=next(c for c in json.loads((Path('runs')/best['run']/'configurations.json').read_text()) if c['name']==best['name'])
_,g=board_spec('T1:val');b=Stage.open(Path('data/T1/E9.5_RNA.h5ad'),g);t=Stage.open(Path('data/T1/E10.5_RNA.h5ad'),g)
ix=np.load('runs/expanded_seed42/indices.npz');used=np.concatenate([ix[k] for k in ['development','ceiling','confirmation','confirmation_ceiling']]);remaining=np.setdiff1d(np.arange(t.n_cells),used);rng.shuffle(remaining);n=len(ix['development']);ref=b.dense(ix['reference']);rows=np.load(Path('runs')/best['run']/'sampled_rows.npz')[best['name']];pred=b.dense(rows);trend,_=load_trend(g)
if cfg['bulk']:pred=apply_trend(pred,trend,cfg['bulk']/5,False,42)
results=[]
for repeat in range(3):
 ids=remaining[2*repeat*n:(2*repeat+1)*n];ci=remaining[(2*repeat+1)*n:(2*repeat+2)*n];target=t.dense(ids);other=t.dense(ci);f=metric_panel(ref,target,ref);c=metric_panel(other,target,ref);m=metric_panel(pred,target,ref);results.append({'repeat':repeat+1,**m,'score':calibrated_score(m,f,c)['local_score']});print(results[-1],flush=True)
write_json(Path('runs/final_audit_seed42.json'),{'selection':best.to_dict(),'configuration':cfg,'fresh_cell_audits':results,'mean_score':float(np.mean([r['score'] for r in results])),'note':'three disjoint target/ceiling pairs; same biological stage; model selected exclusively by development score'})
b.close();t.close()
