"""Validation-selected dense-unit lesions with fresh confirmatory stimuli.

Dense activations are cached once. Replacing activations before the linear
readout gives exactly the same logits as an intervened full forward pass.
No weights are updated and no group is selected using confirmation data.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.special import ndtri
from scipy.stats import rankdata
import torch

from experiment import SmallCNN, generate, TASKS, line, ROOT

NOISES = [.04, .08, .14]
PRIMARY = [.08, .08, .14]


@torch.no_grad()
def cache(model, images, seed, noise=None):
    g = torch.Generator().manual_seed(seed)
    activations = []
    for start in range(0, len(images), 256):
        x = torch.from_numpy(images[start:start+256].copy()).float()[:, None].sub_(128).div_(128)
        if noise is None:
            sd = torch.tensor([.02, .04, .08, .14])[torch.randint(4, (len(x),), generator=g)].view(-1,1,1,1)
        else:
            sd = noise
        x += torch.randn(x.shape, generator=g)*sd
        h = model.classifier[:3](model.features(x))
        activations.append(h.numpy())
    return np.concatenate(activations)


def auc(y, score):
    n1 = int(np.sum(y)); n0 = len(y)-n1
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def margins(logits, labels):
    task = labels//2
    raw = logits[np.arange(len(labels)),task*2+1]-logits[np.arange(len(labels)),task*2]
    return raw*(2*(labels % 2)-1)


def lesion_logits(h, w, b, ids, replacement):
    out = h@w.T+b
    if len(ids):
        out += (replacement[ids]-h[:,ids])@w[:,ids].T
    return out


def masks(meta):
    task=np.array([TASKS.index(m['task']) for m in meta])
    y=np.array([m['label'] for m in meta])
    cond=np.array([m['condition'] for m in meta])
    contrast=np.array([m.get('cue_contrast', -1) for m in meta])
    return dict(task=task, y=y, cond=cond, contrast=contrast)


def scores(logits, labels, meta_masks):
    task=labels//2
    pred=logits.argmax(1)
    binary=(logits[np.arange(len(labels)),task*2+1] > logits[np.arange(len(labels)),task*2]).astype(int)
    correct=binary == labels % 2
    cond=meta_masks['cond']; contrast=meta_masks['contrast']
    accuracy=[]; six=[]; loss=[]; dp=[]
    for t in range(3):
        keep=task==t; y=labels[keep]%2; p=binary[keep]
        accuracy.append(float(correct[keep].mean()))
        six.append(float((pred[keep]==labels[keep]).mean()))
        loss.append(float(np.logaddexp(0,-margins(logits[keep],labels[keep])).mean()))
        hit=((p==1)&(y==1)).sum(); fa=((p==1)&(y==0)).sum()
        # Log-linear correction prevents infinities at ceiling/floor.
        dp.append(float(ndtri((hit+.5)/(sum(y==1)+1))-ndtri((fa+.5)/(sum(y==0)+1))))
    valid=(task==0)&(cond=='valid')&(contrast>0)
    invalid=(task==0)&(cond=='invalid')&(contrast>0)
    hit_v=float(binary[valid].mean()); hit_i=float(binary[invalid].mean())
    search={str(k):float(correct[(task==1)&(cond==f'set_{k}')].mean()) for k in [1,2,4,8]}
    old=float(correct[(task==2)&(cond=='old')].mean())
    new=float(correct[(task==2)&(cond=='new')].mean())
    return dict(accuracy=accuracy, six_class_accuracy=six, cross_entropy=loss, dprime=dp,
                task_identity_accuracy=float((pred//2 == task).mean()),
                effects=dict(posner=hit_v-hit_i,search=search['1']-search['8'],context=old-new),
                hit_valid=hit_v,hit_invalid=hit_i,search_by_set=search,context_old=old,context_new=new), correct


def select_groups(hs, labels, w, b, replacement, active):
    """Greedy loss-based selection on validation ONLY; main k=8 is prespecified."""
    inputs=[]
    for t in range(3):
        use=labels//2==t; h=hs[PRIMARY[t]][use]; y=labels[use]%2
        log=h@w.T+b
        signed=(log[:,t*2+1]-log[:,t*2])*(2*y-1)
        contributions=(h-replacement)*(w[t*2+1]-w[t*2])*(2*y-1)[:,None]
        base_loss=float(np.logaddexp(0,-signed).mean())
        inputs.append((signed,contributions,max(base_loss,.1)))
    groups=[]
    for t in range(3):
        for strategy in ['importance','selective']:
            selected=[]; sums=[np.zeros(len(v[0])) for v in inputs]
            for k in range(1,17):
                candidates=[i for i in active if i not in selected]
                changes=[]
                for signed,contrib,scale in inputs:
                    # Rows are candidates; columns are the same validation trials.
                    lesioned=signed[None,:]-sums[len(changes)][None,:]-contrib[:,candidates].T
                    delta=(np.logaddexp(0,-lesioned).mean(1)-np.logaddexp(0,-signed).mean())/scale
                    changes.append(delta)
                changes=np.stack(changes,axis=1)
                objective=changes[:,t]
                if strategy=='selective':
                    objective=objective-np.maximum(changes[:,[j for j in range(3) if j!=t]],0).mean(1)
                winner=int(candidates[int(np.argmax(objective))]); selected.append(winner)
                for j,(_,contrib,_) in enumerate(inputs): sums[j]+=contrib[:,winner]
                if k in [1,4,8,16]:
                    groups.append(dict(target=TASKS[t],strategy=strategy,k=k,units=selected.copy(),validation_objective=float(objective.max())))
    return groups


def random_accuracy(h, w, b, replacement, groups, labels):
    """Conditional-task accuracy for many random groups using exact margins."""
    out=np.empty((len(groups),3))
    for t in range(3):
        use=labels//2==t; y=labels[use]%2; ht=h[use]
        log=ht@w.T+b
        signed=(log[:,2*t+1]-log[:,2*t])*(2*y-1)
        contrib=(ht-replacement)*(w[2*t+1]-w[2*t])*(2*y-1)[:,None]
        for start in range(0,len(groups),64):
            group_masks=np.zeros((len(groups[start:start+64]),64))
            for i,ids in enumerate(groups[start:start+64]): group_masks[i,ids]=1
            lesioned=signed[None,:]-group_masks@contrib.T
            # Ties: binary rule predicts class 0 (consistent with scores()).
            pred_correct=(lesioned>0)|((lesioned==0)&(y[None,:]==0))
            out[start:start+len(group_masks),t]=pred_correct.mean(1)
    return out


def bootstrap_drop(base_correct, les_correct, labels, meta, seed, n_boot=1500):
    """Paired CI; contextual old trials resampled by repeated configuration."""
    r=np.random.default_rng(seed); out=[]
    for t in range(3):
        use=labels//2==t
        delta=base_correct[use].astype(float)-les_correct[use].astype(float)
        if t!=2:
            # Bootstrap the paired {-1,0,1} differences, preserving dependence
            # between baseline and intervention on each noisy image.
            values,counts=np.unique(delta,return_counts=True)
            draws=r.multinomial(len(delta),counts/counts.sum(),size=n_boot)
            samples=draws@values/len(delta)
        else:
            ctx=[m for m in meta if m['task']=='context']
            old=np.array([m['condition']=='old' for m in ctx])
            configurations=np.array([m['configuration'] for m in ctx])
            old_means=np.array([delta[old & (configurations==c)].mean() for c in sorted(set(configurations[old]))])
            old_boot=old_means[r.integers(len(old_means),size=(n_boot,len(old_means)))].mean(1)
            values,counts=np.unique(delta[~old],return_counts=True)
            new_boot=r.multinomial((~old).sum(),counts/counts.sum(),size=n_boot)@values/(~old).sum()
            samples=.5*(old_boot+new_boot)
        out.append(np.quantile(samples,[.025,.975]).tolist())
    return out


def bh(p):
    p=np.asarray(p); order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    ans=np.empty(len(p));ans[order]=np.minimum(adjusted,1)
    return ans.tolist()


def tuning_probes(model, out):
    """Descriptive AUROC probes, not significance-classified paper neuron types."""
    r=np.random.default_rng(92831); n=512; images=[]; labels=[]
    for side in [0,1]:
        for cue in [0,1]:
            for target in [0,1]:
                for _ in range(n):
                    im=Image.new('L',(192,192),128); d=ImageDraw.Draw(im); x=[14,50][side]
                    line(d,(x,31),r.normal(15 if target else 7,3.5))
                    if cue: d.rectangle([(3*(x-9),63),(3*(x+9),123)],outline=188,width=2)
                    images.append(np.array(im.resize((64,64),Image.Resampling.LANCZOS)))
                    labels.append((side,cue,target,'target'))
        for cue in [0,1]:
            for _ in range(n):
                im=Image.new('L',(192,192),128);d=ImageDraw.Draw(im);x=[14,50][side]
                if cue:d.rectangle([(3*(x-9),63),(3*(x+9),123)],outline=188,width=2)
                images.append(np.array(im.resize((64,64),Image.Resampling.LANCZOS)))
                labels.append((side,cue,0,'cue'))
    h=cache(model,np.stack(images),92832,.08); labels=np.array(labels,dtype=object); rows=[]
    for side in [0,1]:
        for u in range(64):
            vals={}
            for cue in [0,1]:
                idx=(labels[:,0]==side)&(labels[:,1]==cue)&(labels[:,3]=='target')
                vals[f'target_auc_cue_{cue}']=auc(labels[idx,2].astype(int),h[idx,u])
            idx=(labels[:,0]==side)&(labels[:,3]=='cue')
            vals['cue_only_auc']=auc(labels[idx,1].astype(int),h[idx,u])
            vals['cue_change_abs_target_auc']=abs(vals['target_auc_cue_1']-.5)-abs(vals['target_auc_cue_0']-.5)
            rows.append(dict(unit=u,side=['left','right'][side],**vals))
    write_csv(out/'posner_tuning_probes.csv',rows)
    return rows


def write_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--n-val',type=int,default=2000);p.add_argument('--n-test',type=int,default=8000)
    p.add_argument('--random-groups',type=int,default=200);p.add_argument('--out',type=Path,default=ROOT/'lesion_results')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); start=time.time()
    ckpath=ROOT/'results'/'best.pt'; ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    model=SmallCNN();model.load_state_dict(ck['state_dict']);model.eval()
    w=model.classifier[3].weight.detach().numpy().copy();b=model.classifier[3].bias.detach().numpy().copy()
    config=dict(n_val=args.n_val,n_test=args.n_test,random_groups=args.random_groups,
                checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),checkpoint_epoch=ck['epoch'],
                train_seed=142,selection_seed=7142,confirmation_seed=8143,
                train_noise_seed=62001,selection_noise_seed=62002,confirmation_noise_seed=62003,
                noise_levels=NOISES,primary_noise_by_task=dict(zip(TASKS,PRIMARY)),primary_group_size=8,
                primary_mode='mean',primary_strategy='selective',unit_indexing='zero-based; classifier[2] post-ReLU, 64 units',
                python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    print('Caching training means',flush=True)
    train=generate(ck['config']['n_train'],142)
    ht=cache(model,train[0],62001); mu=ht.mean(0); variance=ht.var(0)
    active=np.flatnonzero(variance>1e-8).tolist()
    np.savez_compressed(args.out/'train_activation_stats.npz',mean=mu,variance=variance)
    print('Active dense units',len(active),flush=True)
    if len(active)<16:raise RuntimeError('Fewer than 16 active units; revise planned group sizes before testing')
    del ht, train
    val=generate(args.n_val,7142); test=generate(args.n_test,8143)
    original_test=generate(ck['config']['n_test'],342)
    def signatures(dataset):return {m['configuration'] for m in dataset[2] if m['task']=='context' and m['condition']=='new'}
    train_signatures=signatures(generate(ck['config']['n_train'],142))
    sigs=dict(train=train_signatures,selection=signatures(val),confirmation=signatures(test),previous_test=signatures(original_test))
    overlap={a+'_'+b:len(sigs[a]&sigs[b]) for i,a in enumerate(sigs) for b in list(sigs)[i+1:]}
    assert all(n==0 for n in overlap.values()),overlap
    (args.out/'context_split_audit.json').write_text(json.dumps(overlap,indent=2))
    del original_test
    hv={}; ht={}
    for noise in NOISES:
        hv[noise]=cache(model,val[0],62002,noise)
        ht[noise]=cache(model,test[0],62003,noise)
        print('Cached noise',noise,'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(args.out/'selection_activations.npz',labels=val[1],**{str(n):h for n,h in hv.items()})
    np.savez_compressed(args.out/'confirmation_activations.npz',labels=test[1],**{str(n):h for n,h in ht.items()})
    (args.out/'confirmation_metadata.json').write_text(json.dumps(test[2]))
    # Select all groups before consulting confirmation outcomes.
    groups=[]
    for mode,replacement in [('zero',np.zeros(64)),('mean',mu)]:
        selected=select_groups(hv,val[1],w,b,replacement,active)
        for g in selected:g['mode']=mode
        groups.extend(selected)
    (args.out/'selected_groups.json').write_text(json.dumps(groups,indent=2))
    print('Groups frozen; evaluating confirmation stimuli',flush=True)
    mm=masks(test[2]); baseline={}; base_correct={}; single=[]; group_results=[]; random_results=[]; random_masks={}
    r=np.random.default_rng(62004)
    for k in [1,4,8,16]:
        random_masks[k]=[[u] for u in active] if k==1 else [sorted(r.choice(active,k,replace=False).tolist()) for _ in range(args.random_groups)]
    (args.out/'random_groups.json').write_text(json.dumps(random_masks,indent=2))
    for noise in NOISES:
        bl=ht[noise]@w.T+b
        baseline[str(noise)],base_correct[noise]=scores(bl,test[1],mm)
        for mode,replacement in [('zero',np.zeros(64)),('mean',mu)]:
            for u in range(64):
                logits=lesion_logits(ht[noise],w,b,[u],replacement)
                s,_=scores(logits,test[1],mm)
                record=dict(noise=noise,mode=mode,unit=u)
                for t in range(3):
                    record[f'drop_{TASKS[t]}']=baseline[str(noise)]['accuracy'][t]-s['accuracy'][t]
                    record[f'effect_drop_{TASKS[t]}']=baseline[str(noise)]['effects'][TASKS[t]]-s['effects'][TASKS[t]]
                single.append(record)
            for k in [1,4,8,16]:
                acc=random_accuracy(ht[noise],w,b,replacement,random_masks[k],test[1])
                random_results.append(dict(noise=noise,mode=mode,k=k,accuracy=acc.tolist()))
            for g in groups:
                if g['mode']!=mode:continue
                logits=lesion_logits(ht[noise],w,b,g['units'],replacement)
                s,correct=scores(logits,test[1],mm)
                drops=np.array(baseline[str(noise)]['accuracy'])-np.array(s['accuracy'])
                target=TASKS.index(g['target']);others=[j for j in range(3) if j!=target]
                selectivity=float(drops[target]-max(abs(drops[others])))
                control=next(a for a in random_results if a['noise']==noise and a['mode']==mode and a['k']==g['k'])
                control_drop=np.array(baseline[str(noise)]['accuracy'])[None,:]-np.array(control['accuracy'])
                control_selectivity=control_drop[:,target]-np.abs(control_drop[:,others]).max(1)
                record=dict(**g,noise=noise,scores=s,accuracy_drop=drops.tolist(),selectivity=selectivity,
                    random_drop_mean=control_drop.mean(0).tolist(),
                    random_drop_ci95=np.quantile(control_drop,[.025,.975],axis=0).T.tolist(),
                    random_selectivity_mean=float(control_selectivity.mean()),
                    p_random_selectivity=float((1+sum(control_selectivity>=selectivity))/(len(control_selectivity)+1)),
                    p_random_target_drop=float((1+sum(control_drop[:,target]>=drops[target]))/(len(control_drop)+1)))
                if g['strategy']=='selective' and g['k']==8 and noise==PRIMARY[target]:
                    record['paired_accuracy_drop_ci95']=bootstrap_drop(base_correct[noise],correct,test[1],test[2],62005+target)
                group_results.append(record)
        print('Evaluated noise',noise,'elapsed',round(time.time()-start,1),flush=True)
    primary=[g for g in group_results if g['mode']=='mean' and g['strategy']=='selective' and g['k']==8 and g['noise']==PRIMARY[TASKS.index(g['target'])]]
    for g,q in zip(primary,bh([g['p_random_selectivity'] for g in primary])):g['q_random_selectivity_bh']=q
    results=dict(config=config,active_units=active,baseline=baseline,groups=group_results,primary=primary,
                 elapsed_seconds=time.time()-start)
    (args.out/'results.json').write_text(json.dumps(results,indent=2))
    (args.out/'random_results.json').write_text(json.dumps(random_results))
    write_csv(args.out/'single_unit_lesions.csv',single)
    tuning=[]
    for noise in NOISES:
        for t in range(3):
            use=test[1]//2==t;y=test[1][use]%2
            for u in range(64):tuning.append(dict(noise=noise,task=TASKS[t],unit=u,label_auroc=auc(y,ht[noise][use,u])))
    write_csv(args.out/'task_label_tuning.csv',tuning)
    tuning_probes(model,args.out)
    # Demonstrate equivalence of cached intervention and a full forward hook.
    g=torch.Generator().manual_seed(9971)
    x=torch.tensor(test[0][:32].copy()).float()[:,None].sub(128).div(128)
    x+=torch.randn(x.shape,generator=g)*.08
    with torch.no_grad():
        h=model.classifier[:3](model.features(x)).numpy()
        ids=primary[0]['units']
        def hook(module,inputs,output):
            output=output.clone();output[:,ids]=torch.tensor(mu[ids]);return output
        handle=model.classifier[2].register_forward_hook(hook)
        full=model(x).numpy();handle.remove()
        cached=lesion_logits(h,w,b,ids,mu)
    err=float(np.max(np.abs(full-cached)))
    assert np.allclose(full,cached,atol=2e-5),err
    (args.out/'verification.json').write_text(json.dumps(dict(max_cached_vs_hook_error=err,context_overlap=overlap),indent=2))
    print('DONE',round(time.time()-start,1),'seconds; primary:',flush=True)
    for g in primary:print(g['target'],g['units'],g['accuracy_drop'],'selectivity',g['selectivity'],'q',g['q_random_selectivity_bh'],flush=True)


if __name__=='__main__':main()
