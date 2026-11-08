"""Select lesions for attention-effect reduction while penalizing accuracy loss.

Selection uses cached validation activations only. The selected groups are
frozen to disk before computing confirmation outcomes. This is a secondary
analysis separate from the primary task-selectivity experiment in lesion.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiment import SmallCNN, generate, ROOT, TASKS
from lesion import PRIMARY, masks, scores, lesion_logits, bh, auc


def criterion_check(hv,ht,yv,yt,mv,mt,w,b,ids,replacement,seed):
    """Rule out a mere response-criterion shift in the Posner hit-rate gap.

    The replacement decision threshold is fixed on validation to match the
    baseline validation false-alarm rate; confirmation does not set thresholds.
    AUROC is reported as a threshold-independent descriptive check.
    """
    v=lesion_logits(hv,w,b,[],replacement);vl=lesion_logits(hv,w,b,ids,replacement)
    t=lesion_logits(ht,w,b,[],replacement);tl=lesion_logits(ht,w,b,ids,replacement)
    rawv=v[:,1]-v[:,0];rawvl=vl[:,1]-vl[:,0]
    rawt=t[:,1]-t[:,0];rawtl=tl[:,1]-tl[:,0]
    av=(yv//2==0)&(yv%2==0)&(mv['contrast']>0)
    far=float((rawv[av]>0).mean())
    threshold=float(np.quantile(rawvl[av],1-far,method='inverted_cdf'))
    absent=(yt//2==0)&(yt%2==0)&(mt['contrast']>0)
    valid=(yt//2==0)&(mt['cond']=='valid')&(mt['contrast']>0)
    invalid=(yt//2==0)&(mt['cond']=='invalid')&(mt['contrast']>0)
    pred=rawt>0;les=rawtl>threshold
    baseline_gap=float(pred[valid].mean()-pred[invalid].mean())
    matched_gap=float(les[valid].mean()-les[invalid].mean())
    r=np.random.default_rng(seed);boot=[]
    for keep in [valid,invalid]:
        delta=pred[keep].astype(float)-les[keep].astype(float)
        values,counts=np.unique(delta,return_counts=True)
        boot.append(r.multinomial(len(delta),counts/counts.sum(),size=1500)@values/len(delta))
    aurocs={}
    for name,s in [('baseline',rawt),('lesion',rawtl)]:
        vals=[]
        for keep in [valid,invalid]:
            yy=np.r_[np.ones(keep.sum()),np.zeros(absent.sum())]
            vals.append(auc(yy,np.r_[s[keep],s[absent]]))
        aurocs[name]=dict(valid=vals[0],invalid=vals[1],gap=vals[0]-vals[1])
    return dict(validation_baseline_far=far,validation_selected_threshold=threshold,
                test_baseline_far=float(pred[absent].mean()),test_matched_far=float(les[absent].mean()),
                test_uncalibrated_lesion_far=float((rawtl[absent]>0).mean()),
                baseline_hit_gap=baseline_gap,matched_hit_gap=matched_gap,
                matched_hit_gap_drop=baseline_gap-matched_gap,
                matched_hit_gap_drop_ci95=np.quantile(boot[0]-boot[1],[.025,.975]).tolist(),
                auroc=aurocs)


def fast_effects(h, labels, mm, w, b, replacement, groups, target):
    use=labels//2==target; h=h[use];y=labels[use]%2
    condition=mm['cond'][use];contrast=mm['contrast'][use]
    log=h@w.T+b;raw=log[:,2*target+1]-log[:,2*target]
    contributions=(h-replacement)*(w[2*target+1]-w[2*target])
    masks_group=np.zeros((len(groups),64))
    for j,ids in enumerate(groups):masks_group[j,ids]=1
    raw=raw[None,:]-masks_group@contributions.T
    pred=raw>0;correct=pred==y[None,:]
    acc=correct.mean(1)
    if target==0:
        a=(condition=='valid')&(contrast>0);c=(condition=='invalid')&(contrast>0)
        effect=pred[:,a].mean(1)-pred[:,c].mean(1)
    elif target==1:
        effect=correct[:,condition=='set_1'].mean(1)-correct[:,condition=='set_8'].mean(1)
    else:
        effect=correct[:,condition=='old'].mean(1)-correct[:,condition=='new'].mean(1)
    return effect,acc,correct


def effect_ci(h, labels, mm, meta, w, b, replacement, ids, target,seed):
    base_e,_,base=fast_effects(h,labels,mm,w,b,replacement,[[]],target)
    lesion_e,_,les=fast_effects(h,labels,mm,w,b,replacement,[ids],target)
    use=labels//2==target;cond=mm['cond'][use];contrast=mm['contrast'][use];y=labels[use]%2
    # For hit rate (Posner), correctness on present trials equals positive rate.
    delta=base[0].astype(float)-les[0].astype(float)
    if target==0:
        a=(cond=='valid')&(contrast>0); c=(cond=='invalid')&(contrast>0)
    elif target==1:a=cond=='set_1';c=cond=='set_8'
    else:a=cond=='old';c=cond=='new'
    r=np.random.default_rng(seed);samples=[]
    configs=np.array([m.get('configuration','') for m in meta if m['task']==TASKS[target]])
    for mask in [a,c]:
        if target==2 and np.array_equal(mask,a):
            means=np.array([delta[mask&(configs==name)].mean() for name in sorted(set(configs[mask]))])
            samples.append(means[r.integers(len(means),size=(1500,len(means)))].mean(1))
        else:
            vals,cts=np.unique(delta[mask],return_counts=True)
            samples.append(r.multinomial(mask.sum(),cts/cts.sum(),size=1500)@vals/mask.sum())
    return dict(effect_drop=float(base_e[0]-lesion_e[0]),ci95=np.quantile(samples[0]-samples[1],[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'lesion_results');args=p.parse_args();out=args.out
    config=json.loads((out/'config.json').read_text());rmain=json.loads((out/'results.json').read_text())
    torch.set_num_threads(4)
    model=SmallCNN();model.load_state_dict(torch.load(ROOT/'results'/'best.pt',weights_only=False,map_location='cpu')['state_dict'])
    w=model.classifier[3].weight.detach().numpy();b=model.classifier[3].bias.detach().numpy()
    mu=np.load(out/'train_activation_stats.npz')['mean'];active=rmain['active_units']
    va=np.load(out/'selection_activations.npz');te=np.load(out/'confirmation_activations.npz')
    val=generate(config['n_val'],config['selection_seed']);mv=masks(val[2])
    meta=json.loads((out/'confirmation_metadata.json').read_text());mt=masks(meta)
    groups=[]
    # Criterion fixed before confirmation: reduce the target effect, penalize
    # target accuracy loss twice as much (same units: proportion of trials).
    for mode,replacement in [('mean',mu),('zero',np.zeros(64))]:
        for t in range(3):
            h=va[str(PRIMARY[t])];labels=va['labels']
            be,ba,_=fast_effects(h,labels,mv,w,b,replacement,[[]],t)
            selected=[]
            for k in range(1,9):
                candidates=[selected+[u] for u in active if u not in selected]
                eff,acc,_=fast_effects(h,labels,mv,w,b,replacement,candidates,t)
                effect_drop=be[0]-eff
                acc_drop=ba[0]-acc
                objective=effect_drop-2*np.maximum(acc_drop,0)
                winner=int(np.argmax(objective));selected=candidates[winner]
                if k in [1,4,8]:groups.append(dict(mode=mode,target=TASKS[t],noise=PRIMARY[t],k=k,units=selected.copy(),
                                                validation_effect_drop=float(effect_drop[winner]),validation_accuracy_drop=float(acc_drop[winner]),
                                                validation_objective=float(objective[winner])))
    (out/'attention_selected_groups.json').write_text(json.dumps(dict(accuracy_loss_penalty=2,groups=groups),indent=2))
    random_groups=json.loads((out/'random_groups.json').read_text());results=[]
    for group in groups:
        t=TASKS.index(group['target']);noise=group['noise'];h=te[str(noise)];labels=te['labels']
        replacement=mu if group['mode']=='mean' else np.zeros(64)
        before,bacc,_=fast_effects(h,labels,mt,w,b,replacement,[[]],t)
        after,acc,_=fast_effects(h,labels,mt,w,b,replacement,[group['units']],t)
        re,ra,_=fast_effects(h,labels,mt,w,b,replacement,random_groups[str(group['k'])],t)
        obj=float(before[0]-after[0]-2*max(bacc[0]-acc[0],0))
        robj=before[0]-re-2*np.maximum(bacc[0]-ra,0)
        allscores,_=scores(lesion_logits(h,w,b,group['units'],replacement),labels,mt)
        record=dict(**group,baseline_effect=float(before[0]),lesioned_effect=float(after[0]),
                    baseline_accuracy=float(bacc[0]),lesioned_accuracy=float(acc[0]),
                    effect_drop=float(before[0]-after[0]),accuracy_drop=float(bacc[0]-acc[0]),
                    objective=obj,random_objective_mean=float(robj.mean()),
                    p_random_objective=float((1+sum(robj>=obj))/(len(robj)+1)),scores=allscores)
        if group['k']==8:
            record['paired_effect_drop_ci95']=effect_ci(h,labels,mt,meta,w,b,replacement,group['units'],t,62006+t)['ci95']
            if t==0:
                record['criterion_control']=criterion_check(va[str(noise)],h,va['labels'],labels,mv,mt,w,b,group['units'],replacement,62009)
        results.append(record)
    primary=[g for g in results if g['mode']=='mean' and g['k']==8]
    for g,q in zip(primary,bh([g['p_random_objective'] for g in primary])):g['q_random_objective_bh']=q
    (out/'attention_results.json').write_text(json.dumps(dict(accuracy_loss_penalty=2,groups=results,primary=primary),indent=2))
    for g in primary: print(g['target'],g['units'],'effect drop',g['effect_drop'],'accuracy drop',g['accuracy_drop'],'q',g['q_random_objective_bh'],flush=True)


if __name__=='__main__':main()
