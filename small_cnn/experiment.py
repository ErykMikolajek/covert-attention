"""Small joint CNN for three covert attention tasks. Run from repository root.

Synthetic stimuli follow the local notebooks and mmc2 methods, with explicitly
documented adaptations; this is not an exact reproduction of the published CNN.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
TASKS = ['posner', 'search', 'context']
SIZE = 64


def old_contexts():
    r = np.random.default_rng(8142)
    return [dict(target=t, distractors=r.choice([x for x in range(16) if x != t], 7, replace=False).tolist(),
                 rotations=r.integers(0, 4, 7).tolist()) for t in range(16)]


def line(draw, xy, angle, length=16, value=151):
    a = math.radians(angle)
    dx, dy = length / 2 * math.sin(a), length / 2 * math.cos(a)
    x, y = xy
    draw.line([(round((x-dx)*3), round((y+dy)*3)),
               (round((x+dx)*3), round((y-dy)*3))], fill=value, width=3)


def glyph(draw, xy, kind, rotation):
    # T points left/right (90/270 deg); L has four cardinal rotations.
    points = [(-3.5, -3.5), (3.5, -3.5), (0, -3.5), (0, 3.5)] if kind == 'T' else [(-3.5, -3.5), (-3.5, 3.5), (3.5, 3.5)]
    a = math.radians(rotation)
    pts = [(round(3*(xy[0]+x*math.cos(a)-y*math.sin(a))),
            round(3*(xy[1]+x*math.sin(a)+y*math.cos(a)))) for x,y in points]
    if kind == 'T':
        draw.line(pts[:2], fill=160, width=3)
        draw.line(pts[2:], fill=160, width=3)
    else:
        draw.line(pts, fill=160, width=3)


def generate(n_per_task, seed):
    r = np.random.default_rng(seed)
    xs, ys, meta = [], [], []
    old = old_contexts()
    for task in range(3):
        for i in range(n_per_task):
            y = i % 2
            im = Image.new('L', (SIZE*3, SIZE*3), 128)
            d = ImageDraw.Draw(im)
            m = dict(task=TASKS[task], label=y)
            if task == 0:
                cue = int(r.integers(2))
                valid = bool(r.random() < .8)
                target = cue if valid else 1-cue
                contrast = int(r.choice([0, 20, 40, 60]))
                for j,x in enumerate([14,50]):
                    angle = r.normal(15 if y and j == target else 7, 3.5)
                    line(d,(x,31),angle)
                x = [14,50][cue]
                d.rectangle([(3*(x-9),3*21),(3*(x+9),3*41)], outline=128+contrast, width=2)
                m.update(condition='valid' if y and valid else 'invalid' if y else 'absent', cue=cue,
                         target=target if y else -1, cue_contrast=contrast)
            elif task == 1:
                ss = [1,2,4,8][(i//2)%4]
                if ss == 1:
                    cued = [int(r.integers(8))]
                elif ss == 2:
                    a = int(r.integers(4)); cued = [a,a+4]
                elif ss == 4:
                    cued = list(range(int(r.integers(2)),8,2))
                else:
                    cued = list(range(8))
                target = int(r.choice(cued))
                for j in range(8):
                    a = j*math.pi/4
                    x,yc = 32+23*math.sin(a),32-23*math.cos(a)
                    line(d,(x,yc),15 if y and j==target else 0,length=9)
                    if j in cued:
                        d.rectangle([(round(3*(x-5)),round(3*(yc-6))),
                                     (round(3*(x+5)),round(3*(yc+6)))],outline=180,width=2)
                m.update(condition=f'set_{ss}', set_size=ss, target=target if y else -1)
            else:
                is_old = (i//2)%2 == 0
                if is_old:
                    c = old[(i//4)%16]
                else:
                    t = int(r.integers(16))
                    c = dict(target=t, distractors=r.choice([x for x in range(16) if x!=t],7,replace=False).tolist(),
                             rotations=r.integers(0,4,7).tolist())
                positions = [(9+15*(j%4),9+15*(j//4)) for j in range(16)]
                glyph(d,positions[c['target']],'T',90 if y else 270)
                for j,rot in zip(c['distractors'],c['rotations']):
                    glyph(d,positions[j],'L',int(rot)*90)
                signature = json.dumps(c,sort_keys=True)
                m.update(condition='old' if is_old else 'new', target=c['target'],
                         configuration=hashlib.sha256(signature.encode()).hexdigest())
            im = im.resize((SIZE,SIZE),Image.Resampling.LANCZOS)
            xs.append(np.asarray(im)); ys.append(task*2+y); meta.append(m)
    return np.stack(xs), np.array(ys,dtype=np.int64), meta


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
                                      nn.Conv2d(16,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                      nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(32*8*8,64),nn.ReLU(),nn.Linear(64,6))

    def forward(self,x):
        return self.classifier(self.features(x))


def wilson(k,n):
    if not n: return [None,None]
    p=k/n; z=1.96; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


@torch.no_grad()
def predictions(model,x,noise,seed):
    g=torch.Generator().manual_seed(seed)
    outs=[]
    model.eval()
    for b in x.split(256):
        v=b+torch.randn(b.shape,generator=g)*noise
        outs.append(model(v))
    return torch.cat(outs)


def metrics(logits,labels,meta):
    pred=logits.argmax(1).numpy(); labels=np.asarray(labels)
    rows=[]
    for task in TASKS:
        conditions=['all']+sorted({m['condition'] for m in meta if m['task']==task})
        for c in conditions:
            idx=np.array([i for i,m in enumerate(meta) if m['task']==task and (c=='all' or m['condition']==c)])
            correct=int((pred[idx]==labels[idx]).sum()); n=len(idx)
            t=TASKS.index(task); local=logits[idx,t*2:t*2+2].argmax(1).numpy()
            row=dict(task=task,condition=c,n=n,accuracy=correct/n,ci95=wilson(correct,n),
                     task_given_accuracy=float((local==labels[idx]%2).mean()))
            if task!='context':
                pos=idx[labels[idx]%2==1]; neg=idx[labels[idx]%2==0]
                row['hit_rate']=float((pred[pos]==t*2+1).mean()) if len(pos) else None
                row['false_alarm_rate']=float((pred[neg]==t*2+1).mean()) if len(neg) else None
            rows.append(row)
    return rows


def save_preview(x,meta,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,6,figsize=(12,6))
    for t in range(3):
        ids=[i for i,m in enumerate(meta) if m['task']==TASKS[t]][:6]
        for ax,i in zip(axs[t],ids):
            ax.imshow(x[i],cmap='gray',vmin=80,vmax=185)
            ax.set_title(f"{TASKS[t]} {meta[i]['condition']}\ny={meta[i]['label']}",fontsize=8); ax.axis('off')
    fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)


def audit(splits):
    report={}
    names=list(splits)
    hashes={name:set(hashlib.sha256(a.tobytes()).hexdigest() for a in splits[name][0]) for name in names}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        # Shared clean old contexts are intentional; noisy images are independently sampled.
        new_a={m['configuration'] for m in splits[a][2] if m['task']=='context' and m['condition']=='new'}
        new_b={m['configuration'] for m in splits[b][2] if m['task']=='context' and m['condition']=='new'}
        assert not (new_a & new_b), 'New context leakage'
        report[a+'_'+b]=dict(shared_clean_images=len(hashes[a]&hashes[b]),shared_new_contexts=len(new_a&new_b))
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--n-train',type=int,default=10000); p.add_argument('--n-val',type=int,default=2000)
    p.add_argument('--n-test',type=int,default=4000); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=ROOT/'results'); p.add_argument('--evaluate',action='store_true')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    config=vars(args).copy(); config['out']=str(config['out']); config.update(image_size=SIZE,
        train_noise=[.02,.04,.08,.14],validation_noise=.04,labels=['posner_absent','posner_present','search_absent','search_present','context_left','context_right'])
    splits={k:generate(n,args.seed+s) for k,n,s in [('train',args.n_train,100),('val',args.n_val,200),('test',args.n_test,300)]}
    audit_report=audit(splits)
    (args.out/'split_audit.json').write_text(json.dumps(audit_report,indent=2))
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    save_preview(splits['test'][0],splits['test'][2],args.out/'stimuli.png')
    xs={k:torch.from_numpy(v[0].copy()).float().unsqueeze(1).sub_(128).div_(128) for k,v in splits.items()}
    ys={k:torch.from_numpy(v[1]) for k,v in splits.items()}
    model=SmallCNN(); count=sum(p.numel() for p in model.parameters()); print('Parameters:',count,flush=True)
    start=time.time(); history=[]; best=-1
    if not args.evaluate:
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.0001)
        for epoch in range(args.epochs):
            model.train(); order=torch.randperm(len(xs['train'])); total=0; acc=0
            for ids in order.split(128):
                b=xs['train'][ids]
                noise=torch.tensor(config['train_noise'])[torch.randint(4,(len(ids),))].view(-1,1,1,1)
                b=b+torch.randn_like(b)*noise
                opt.zero_grad(set_to_none=True); logits=model(b); loss=nn.functional.cross_entropy(logits,ys['train'][ids]); loss.backward(); opt.step()
                total+=float(loss)*len(ids); acc+=int((logits.argmax(1)==ys['train'][ids]).sum())
            sched.step()
            val=predictions(model,xs['val'],.04,913)
            score=float((val.argmax(1)==ys['val']).float().mean())
            entry=dict(epoch=epoch+1,loss=total/len(order),train_accuracy=acc/len(order),val_accuracy=score,seconds=time.time()-start)
            history.append(entry); print(json.dumps(entry),flush=True)
            if score>best:
                best=score; torch.save(dict(state_dict=model.state_dict(),config=config,parameters=count,epoch=epoch+1),args.out/'best.pt')
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
    checkpoint=torch.load(args.out/'best.pt',weights_only=False,map_location='cpu'); model.load_state_dict(checkpoint['state_dict'])
    saved_history=history or (json.loads((args.out/'history.json').read_text()) if (args.out/'history.json').exists() else [])
    result=dict(parameters=count,best_epoch=checkpoint['epoch'],training_seconds=saved_history[-1]['seconds'] if saved_history else None,evaluations={})
    all_predictions=[]
    for noise in [0.,.04,.08,.14,.20]:
        logits=predictions(model,xs['test'],noise,1913)
        result['evaluations'][str(noise)]=metrics(logits,ys['test'],splits['test'][2])
        probs=logits.softmax(1).numpy()
        for i,m in enumerate(splits['test'][2]):
            all_predictions.append(dict(index=i,noise=noise,**m,prediction=int(logits[i].argmax()),p_class_1=float(probs[i,TASKS.index(m['task'])*2+1])))
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (args.out/'test_predictions.jsonl').open('w') as f:
        for row in all_predictions: f.write(json.dumps(row)+'\n')
    for row in result['evaluations']['0.04']: print(row,flush=True)
    # Raw clean stimuli are reproducible; store held-out test data for direct inspection.
    np.savez_compressed(args.out/'test_clean.npz',images=splits['test'][0],labels=splits['test'][1])
    (args.out/'test_metadata.json').write_text(json.dumps(splits['test'][2]))


if __name__=='__main__': main()
