"""Apply each frozen k=8 group using BOTH replacement rules (same unit IDs)."""
import json
import numpy as np
import torch
from experiment import ROOT, SmallCNN, TASKS, generate
from lesion import masks, lesion_logits, scores, PRIMARY
from lesion_attention import criterion_check


def main():
    out=ROOT/'lesion_results'
    r=json.loads((out/'results.json').read_text());a=json.loads((out/'attention_results.json').read_text())
    config=r['config'];model=SmallCNN()
    model.load_state_dict(torch.load(ROOT/'results'/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
    w=model.classifier[3].weight.detach().numpy();b=model.classifier[3].bias.detach().numpy()
    mu=np.load(out/'train_activation_stats.npz')['mean'];te=np.load(out/'confirmation_activations.npz')
    va=np.load(out/'selection_activations.npz');meta=json.loads((out/'confirmation_metadata.json').read_text());mm=masks(meta)
    mv=masks(generate(config['n_val'],config['selection_seed'])[2])
    rows=[]
    for family,groups in [('task',r['groups']),('attention',a['groups'])]:
        for g in groups:
            t=TASKS.index(g['target'])
            if g['k']!=8 or g['noise']!=PRIMARY[t]:continue
            if family=='task' and g['strategy']!='selective':continue
            for mode,replacement in [('mean',mu),('zero',np.zeros(64))]:
                h=te[str(g['noise'])];labels=te['labels']
                s,_=scores(lesion_logits(h,w,b,g['units'],replacement),labels,mm)
                baseline=r['baseline'][str(g['noise'])]
                row=dict(family=family,target=g['target'],selection_mode=g['mode'],intervention_mode=mode,
                         noise=g['noise'],units=g['units'],scores=s,
                         accuracy_drop=(np.array(baseline['accuracy'])-s['accuracy']).tolist(),
                         effect_drop=baseline['effects'][g['target']]-s['effects'][g['target']])
                if family=='attention' and t==0:
                    row['criterion_control']=criterion_check(va[str(g['noise'])],h,va['labels'],labels,mv,mm,w,b,g['units'],replacement,62009)
                rows.append(row)
    (out/'cross_replacement_checks.json').write_text(json.dumps(rows,indent=2))
    for row in rows:
        if row['selection_mode']=='mean':
            print(row['family'],row['target'],row['intervention_mode'],row['accuracy_drop'],row['effect_drop'])


if __name__=='__main__':main()
