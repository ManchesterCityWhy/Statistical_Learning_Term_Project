"""Training-stage-only candidates; fixed development/confirmation draw split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from t1_pipeline.data import Stage,board_spec,write_json
from t1_pipeline.evaluation import metric_panel,calibrated_score
from experiments.bulk_trend import load_trend,apply_trend
from experiments.geo_transport import renormalize

def main():
 out=Path('runs/expanded_seed42');out.mkdir(exist_ok=False,parents=True)
 rng=np.random.default_rng(42)
 _,genes=board_spec('T1:val')
 a,b,t=[Stage.open(Path(f'data/T1/{s}_RNA.h5ad'),genes) for s in ['E8.5','E9.5','E10.5']]
 trend,external=load_trend(genes)
 # Evaluation partitions are fixed before candidates are evaluated; confirm is not searched.
 order=rng.permutation(t.n_cells);n=round(.1*t.n_cells)//2
 target=t.dense(order[:n]);ceiling=t.dense(order[n:2*n])
 confirm=t.dense(order[2*n:3*n]);confirm_ceiling=t.dense(order[3*n:4*n])
 rows=rng.choice(b.n_cells,1706,replace=False);ref=b.dense(rows)
 np.savez_compressed(out/'indices.npz',reference=rows,development=order[:n],ceiling=order[n:2*n],confirmation=order[2*n:3*n],confirmation_ceiling=order[3*n:4*n])
 floor=metric_panel(ref,target,ref);ceil=metric_panel(ceiling,target,ref)
 labels=b.labels;la=a.labels
 # Harmonize training labels only, joining closely related cardiac lineages.
 def group(x):
  if 'CM' in x:return 'Cardiomyocyte'
  if x in ['Endothelium','Endocardium','BEC']:return 'Endothelial'
  if x in ['NCC','NCC-derived']:return 'Neural crest'
  return x
 ga=np.array([group(x) for x in la]);gb=np.array([group(x) for x in labels])
 deltas={};ratios={}
 for label in sorted(set(ga)&set(gb)):
  ma=a.mean(np.flatnonzero(ga==label));mb=b.mean(np.flatnonzero(gb==label))
  deltas[label]=mb-ma
  ratios[label]=np.clip(np.log((mb+.05)/(ma+.05)),-1,1)
 candidates=[{'name':'copy_last','type':0,'bulk':0,'kind':'add'}]
 for kind in ['add','ratio']:
  for typ in [0,.5,1,2]:
   if kind=='ratio' and typ==0:continue
   for bulk in [0,.75,1.5,3,5]:
    candidates.append({'name':f'{kind}_type{typ}_bulk{bulk}','type':typ,'bulk':bulk,'kind':kind})
 def predict(c):
  x=ref.copy()
  if c['type']:
   for lab in deltas:
    mask=gb[rows]==lab
    if c['kind']=='add':x[mask]=np.maximum(x[mask]+c['type']*deltas[lab]*(x[mask]>0),0)
    else:x[mask]=np.log1p(np.expm1(x[mask])*np.exp(c['type']*ratios[lab]))
   x=renormalize(x)
  if c['bulk']:x=apply_trend(x,trend,c['bulk']/5,False,42)
  return x
 results=[]
 for c in candidates:
  m=metric_panel(predict(c),target,ref)
  results.append({**c,**m,**{'score':calibrated_score(m,floor,ceil)['local_score']}})
  pd.DataFrame(results).to_csv(out/'development.csv',index=False)
  print(c['name'],results[-1]['score'],flush=True)
 selected=sorted(results,key=lambda x:x['score'],reverse=True)[:3]
 cf=metric_panel(ref,confirm,ref);cc=metric_panel(confirm_ceiling,confirm,ref)
 checks=[]
 for c in [candidates[0],*selected]:
  m=metric_panel(predict(c),confirm,ref);checks.append({'name':c['name'],**m,'score':calibrated_score(m,cf,cc)['local_score']})
 pd.DataFrame(checks).to_csv(out/'confirmation.csv',index=False)
 write_json(out/'selection.json',{'seed':42,'external':external,'selected_on_development':selected,'confirmation':checks,'status':'complete','note':'confirmation cells disjoint; same embryo stages, not independent biological replication'})
 for s in [a,b,t]:s.close()
if __name__=='__main__':main()
