"""Protocol checks: balancing, repeated-context semantics and held-out novelty."""
import unittest
import numpy as np
import torch
from experiment import SmallCNN, generate, audit, old_contexts


class ProtocolTests(unittest.TestCase):
    def test_balancing_and_reproducibility(self):
        x,y,m=generate(128,123)
        x2,y2,m2=generate(128,123)
        self.assertTrue(np.array_equal(x,x2))
        self.assertTrue(np.array_equal(y,y2))
        self.assertEqual(m,m2)
        self.assertEqual(np.bincount(y).tolist(),[64]*6)
        for c in ['set_1','set_2','set_4','set_8','old','new']:
            labels=[a['label'] for a in m if a['condition']==c]
            self.assertEqual(labels.count(0),labels.count(1))

    def test_old_contexts_do_not_predict_orientation(self):
        _,_,meta=generate(256,123)
        old=[m for m in meta if m['task']=='context' and m['condition']=='old']
        for signature in {m['configuration'] for m in old}:
            labels=[m['label'] for m in old if m['configuration']==signature]
            self.assertEqual(labels.count(0),labels.count(1))
        self.assertEqual(len(old_contexts()),16)
        self.assertEqual({c['target'] for c in old_contexts()},set(range(16)))

    def test_new_contexts_are_held_out(self):
        report=audit({k:generate(256,s) for k,s in [('train',1),('val',2),('test',3)]})
        self.assertTrue(all(v['shared_new_contexts']==0 for v in report.values()))

    def test_single_forward_six_labels(self):
        model=SmallCNN()
        self.assertEqual(tuple(model(torch.zeros(2,1,64,64)).shape),(2,6))
        self.assertLess(sum(p.numel() for p in model.parameters()),150000)


if __name__=='__main__': unittest.main()
