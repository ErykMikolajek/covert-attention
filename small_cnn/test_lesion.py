"""Intervention checks independent of task performance outcomes."""
import unittest
import numpy as np
import torch
from experiment import SmallCNN
from lesion import lesion_logits, bh
from lesion_readout_controls import best_threshold


class LesionTests(unittest.TestCase):
    def test_cached_lesion_matches_full_forward(self):
        torch.manual_seed(971);model=SmallCNN().eval()
        x=torch.randn(7,1,64,64);ids=[0,4,17,63]
        replacement=np.linspace(-.1,.1,64).astype(np.float32)
        with torch.no_grad():
            h=model.classifier[:3](model.features(x)).numpy()
            w=model.classifier[3].weight.numpy(); b=model.classifier[3].bias.numpy()
            def hook(module,inputs,output):
                y=output.clone();y[:,ids]=torch.tensor(replacement[ids]);return y
            handle=model.classifier[2].register_forward_hook(hook)
            full=model(x).numpy();handle.remove()
        np.testing.assert_allclose(lesion_logits(h,w,b,ids,replacement),full,atol=1e-6)

    def test_full_lesion_removes_image_information(self):
        r=np.random.default_rng(71)
        h=r.normal(size=(13,64));w=r.normal(size=(6,64));b=r.normal(size=6);mu=r.normal(size=64)
        actual=lesion_logits(h,w,b,list(range(64)),mu)
        np.testing.assert_allclose(actual,np.tile(mu@w.T+b,(13,1)),atol=1e-12)
        np.testing.assert_array_equal(lesion_logits(h,w,b,[],mu),h@w.T+b)

    def test_multiple_comparison_correction(self):
        np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04])

    def test_threshold_control_recovers_shifted_decisions(self):
        # Pure score offset destroys fixed-zero accuracy but not information.
        score=np.array([8.,9.,11.,12.]);y=np.array([0,0,1,1])
        self.assertEqual(float(((score>0)==y).mean()),.5)
        threshold=best_threshold(score,y)
        self.assertTrue(np.array_equal(score>threshold,y))


if __name__=='__main__':unittest.main()
