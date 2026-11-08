"""Separate representational information loss from readout/criterion changes.

Diagnostic only: original groups remain frozen. Evaluate AUROC, validation-only
threshold recalibration, and replacement by within-task training means.
"""
import json
import numpy as np
import torch
from experiment import ROOT, SmallCNN, generate, TASKS
from lesion import cache, auc, lesion_logits, scores, masks


def best_threshold(score,y):
    order=np.argsort(score,kind='stable');s=score[order];yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    correct=y.sum()+np.cumsum(1-2*yy)[ends]
    candidates=np.r_[y.sum(),correct]
    thresholds=np.r_[np.nextafter(s[0],-np.inf),s[ends]]
    return float(thresholds[np.argmax(candidates)])


def main():
    out=ROOT/'lesion_results';r=json.loads((out/'results.json').read_text());config=r['config']
    torch.set_num_threads(4);model=SmallCNN();ck=torch.load(ROOT/'results'/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict']);model.eval()
    w=model.classifier[3].weight.detach().numpy();b=model.classifier[3].bias.detach().numpy()
    mu=np.load(out/'train_activation_stats.npz')['mean']
    train=generate(ck['config']['n_train'],142);htrain=cache(model,train[0],config['train_noise_seed'])
    task_mu=np.stack([htrain[train[1]//2==t].mean(0) for t in range(3)])
    np.savez_compressed(out/'within_task_train_means.npz',mean=task_mu)
    te=np.load(out/'confirmation_activations.npz');va=np.load(out/'selection_activations.npz')
    meta=json.loads((out/'confirmation_metadata.json').read_text());mm=masks(meta)
    rows=[]
    for group in r['primary']:
        noise=group['noise'];ids=group['units'];hv=va[str(noise)];ht=te[str(noise)]
        basev=hv@w.T+b;baset=ht@w.T+b
        lesv=lesion_logits(hv,w,b,ids,mu);lest=lesion_logits(ht,w,b,ids,mu)
        conditional=baset.copy()
        for t in range(3):
            use=te['labels']//2==t
            conditional[use]=lesion_logits(ht[use],w,b,ids,task_mu[t])
        condscores,_=scores(conditional,te['labels'],mm)
        tests=[]
        for t in range(3):
            iv=va['labels']//2==t;it=te['labels']//2==t
            yv=va['labels'][iv]%2;yt=te['labels'][it]%2
            sv=basev[iv,2*t+1]-basev[iv,2*t];slv=lesv[iv,2*t+1]-lesv[iv,2*t]
            st=baset[it,2*t+1]-baset[it,2*t];slt=lest[it,2*t+1]-lest[it,2*t]
            tb=best_threshold(sv,yv);tl=best_threshold(slv,yv)
            tests.append(dict(task=TASKS[t],baseline_auroc=auc(yt,st),lesioned_auroc=auc(yt,slt),
                              baseline_threshold=tb,lesion_threshold=tl,
                              baseline_calibrated_accuracy=float(((st>tb)==yt).mean()),
                              lesion_calibrated_accuracy=float(((slt>tl)==yt).mean()),
                              calibrated_accuracy_drop=float(((st>tb)==yt).mean()-((slt>tl)==yt).mean())))
        rows.append(dict(target=group['target'],units=ids,noise=noise,threshold_checks=tests,
                         within_task_mean_scores=condscores,
                         within_task_mean_accuracy_drop=(np.array(r['baseline'][str(noise)]['accuracy'])-condscores['accuracy']).tolist()))
    (out/'readout_controls.json').write_text(json.dumps(rows,indent=2))
    for g in rows:
        target=next(t for t in g['threshold_checks'] if t['task']==g['target'])
        print(g['target'],'AUC',target['baseline_auroc'],target['lesioned_auroc'],
              'recalibrated accuracy',target['baseline_calibrated_accuracy'],target['lesion_calibrated_accuracy'],
              'within-task mean drops',g['within_task_mean_accuracy_drop'],flush=True)


if __name__=='__main__':main()
