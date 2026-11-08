"""Optimistic Posner benchmark with noiseless access to latent angles and cue.

This is not the image-computable Bayesian observer from the paper. It ignores
rasterization/luminance noise and assumes the cue side is known even at contrast 0.
It quantifies the irreducible ambiguity caused by overlapping angle distributions.
"""
import json
from pathlib import Path
import numpy as np

r=np.random.default_rng(9142)
n=1_000_000
y=r.integers(2,size=n)
target=(r.random(n)>=.8).astype(int)  # cue defines location 0
a=r.normal(7,3.5,(n,2))
a[np.arange(n),target]+=8*y
# f_target(a)/f_distractor(a) for equal-variance Gaussians.
lr=np.exp(8/(3.5**2)*(a-11))
bayes_factor=lr@np.array([.8,.2])
result=dict(n=n,seed=9142,latent_angle_oracle_accuracy=float(((bayes_factor>1)==y).mean()),
            caveat='Optimistic latent-angle benchmark, not image BIO; assumes fully observed cue and angles.')
out=Path(__file__).resolve().parent/'results'/'angle_oracle.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
