"""Inference on a generated test sample or an externally supplied stimulus.

Example: python3 small_cnn/predict.py --index 0 --noise .04
Input images should have gray background 128 and geometry matching the generator.
"""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from experiment import SmallCNN, ROOT

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default=str(ROOT/'results'/'best.pt'))
p.add_argument('--image')
p.add_argument('--index',type=int,default=0)
p.add_argument('--noise',type=float,default=0.)
p.add_argument('--task',choices=['posner','search','context'])
a=p.parse_args()
torch.set_num_threads(4)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=SmallCNN(); model.load_state_dict(ck['state_dict']); model.eval()
if a.image:
    x=np.array(Image.open(a.image).convert('L').resize((64,64),Image.Resampling.LANCZOS))
else:
    x=np.load(ROOT/'results'/'test_clean.npz')['images'][a.index]
x=torch.tensor(x.copy(),dtype=torch.float32)[None,None].sub(128).div(128)
torch.manual_seed(999); x+=torch.randn_like(x)*a.noise
with torch.no_grad():
    logits=model(x)[0]
    if a.task:
        t=['posner','search','context'].index(a.task)
        probs=logits[t*2:t*2+2].softmax(0)
        labels=ck['config']['labels'][t*2:t*2+2]
    else:
        probs=logits.softmax(0); labels=ck['config']['labels']
print(json.dumps(dict(prediction=labels[int(probs.argmax())],probabilities=dict(zip(labels,probs.tolist()))),indent=2))
