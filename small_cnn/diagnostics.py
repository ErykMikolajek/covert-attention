"""Additional stratification using already saved held-out predictions; no fitting."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

R=Path(__file__).resolve().parent/'results'
data=[json.loads(line) for line in (R/'test_predictions.jsonl').open()]
rows=[]
for noise in [0.,.04,.08,.14,.20]:
    for contrast in [0,20,40,60]:
        for cond in ['valid','invalid','absent']:
            sample=[a for a in data if a['noise']==noise and a['task']=='posner' and a['cue_contrast']==contrast and a['condition']==cond]
            rows.append(dict(noise=noise,cue_contrast=contrast,condition=cond,n=len(sample),
                             positive_rate=sum(a['prediction']==1 for a in sample)/len(sample)))
with (R/'posner_by_contrast.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

summary={}
for noise in [0.,.04,.08,.14,.20]:
    values={}
    for task in ['posner','search','context']:
        sample=[a for a in data if a['noise']==noise and a['task']==task]
        y=np.array([a['label'] for a in sample]); score=np.array([a['p_class_1'] for a in sample])
        ranks=rankdata(score); n1=int(y.sum());n0=len(y)-n1
        values[task]=dict(auroc=float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)))
    cm=np.zeros((6,6),dtype=int)
    for a in data:
        if a['noise']==noise:
            t=['posner','search','context'].index(a['task']); cm[t*2+a['label'],a['prediction']]+=1
    values['confusion_matrix']=cm.tolist();summary[str(noise)]=values
(R/'diagnostics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['0.04'],indent=2))
