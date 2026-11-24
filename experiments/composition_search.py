"""Estimate composition from permitted bulk trend, never target proportions."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from t1_pipeline.data import Stage,board_spec,write_json
from t1_pipeline.evaluation import metric_panel,calibrated_score
from experiments.bulk_trend import load_trend,apply_trend

def main():
 out=Path('runs/composition_seed42');out.mkdir(parents=True,exist_ok=False)
 rng=np.random.default_rng(42);_,genes=board_spec('T1:val')
 b=Stage.open(Path('data/T1/E9.5_RNA.h5ad'),genes);t=Stage.open(Path('data/T1/E10.5_RNA.h5ad'),genes)
 ix=np.load('runs/expanded_seed42/indices.npz');ref=b.dense(ix['reference']);target=t.dense(ix['development']);other=t.dense(ix['ceiling'])
 floor=metric_panel(ref,target,ref);ceil=metric_panel(other,target,ref)
 trend,ext=load_trend(genes);labels=b.labels;types=np.unique(labels)
 means=[];p=[]
 for lab in types:
  ids=np.flatnonzero(labels==lab);p.append(len(ids)/b.n_cells);total=np.zeros(len(genes))
  for chunk in np.array_split(ids,max(1,len(ids)//128)):total+=np.expm1(b.dense(chunk)).sum(axis=0,dtype=np.float64)
  means.append(total/len(ids))
 profiles=np.stack(means,axis=1);p=np.array(p);base=profiles@p
 # Strongly expressed marker genes stabilize relative bulk deconvolution.
 keep=(base>.1)&(np.std(profiles,axis=1)>.1)
 design=profiles[keep]/np.sqrt(base[keep,None]+1)
 records=[];configs=[]
 for strength in [.25,.5,1,2]:
  desired=base*np.exp(np.clip(trend*strength,-2,2));desired*=base.sum()/desired.sum()
  # Ridge toward the observed composition bounds unsupported changes.
  y=desired[keep]/np.sqrt(base[keep]+1)
  q=nnls(np.vstack([design,10*np.eye(len(p))]),np.r_[y,10*p])[0];q/=q.sum()
  for blend in [.2,.5,1]:
   prop=(1-blend)*p+blend*q
   weights=np.zeros(b.n_cells)
   for lab,prob in zip(types,prop):weights[labels==lab]=prob/(labels==lab).sum()
   rows=rng.choice(b.n_cells,1706,replace=False,p=weights/weights.sum())
   for bulk in [0,1.5,5]:
    configs.append({'name':f'deconv{strength}_mix{blend}_bulk{bulk}','strength':strength,'blend':blend,'bulk':bulk,'rows':rows,'proportions':dict(zip(types,prop))})
 for c in configs:
  x=b.dense(c['rows'])
  if c['bulk']:x=apply_trend(x,trend,c['bulk']/5,False,42)
  m=metric_panel(x,target,ref);records.append({'name':c['name'],**m,'score':calibrated_score(m,floor,ceil)['local_score']})
  pd.DataFrame(records).to_csv(out/'development.csv',index=False);print(c['name'],records[-1]['score'],flush=True)
 best=sorted(records,key=lambda r:r['score'],reverse=True)[:3];confirm=t.dense(ix['confirmation']);co=t.dense(ix['confirmation_ceiling']);cf=metric_panel(ref,confirm,ref);cc=metric_panel(co,confirm,ref);checks=[]
 for row in best:
  c=next(c for c in configs if c['name']==row['name']);x=b.dense(c['rows'])
  if c['bulk']:x=apply_trend(x,trend,c['bulk']/5,False,42)
  m=metric_panel(x,confirm,ref);checks.append({'name':c['name'],**m,'score':calibrated_score(m,cf,cc)['local_score']})
 pd.DataFrame(checks).to_csv(out/'confirmation.csv',index=False)
 write_json(out/'configurations.json',[{k:v for k,v in c.items() if k!='rows'} for c in configs]);np.savez_compressed(out/'sampled_rows.npz',**{c['name']:c['rows'] for c in configs})
 b.close();t.close()
if __name__=='__main__':main()
