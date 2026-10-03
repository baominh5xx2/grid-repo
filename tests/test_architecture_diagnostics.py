import unittest
import torch
from torch import nn
from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT


class ArchitectureDiagnosticsTests(unittest.TestCase):
    def test_entropy_ignores_padding_and_empty_singleton_segments(self):
        weights = torch.tensor([[.5,.5,0.], [1.,0.,0.], [0.,0.,0.]])
        mask = torch.tensor([[True,True,False],[True,False,False],[False,False,False]])
        entropy = FlatCafeBERT._normalized_attention_entropy(weights, mask)
        torch.testing.assert_close(entropy, torch.tensor([1.,0.,0.]))

    def test_gate_diagnostics_measure_channels_and_branch_norm_without_graph(self):
        model = FlatCafeBERT.__new__(FlatCafeBERT)
        nn.Module.__init__(model)
        model.use_gate = True
        model.fusion_gate = nn.Linear(8,4)
        model.fusion_norm = nn.LayerNorm(4)
        model.eval()
        c = torch.randn(2,4,requires_grad=True)
        r = torch.randn(2,4,requires_grad=True)
        output = model._residual_fuse(c,r)
        d = model.last_architecture_diagnostics
        self.assertEqual(d['gate_channel_std'].shape,(2,))
        self.assertFalse(d['relation_to_cls_norm_ratio'].requires_grad)
        gate = torch.sigmoid(model.fusion_gate(torch.cat([c,r],-1)))
        expected = (gate * model.fusion_norm(r)).float().norm(dim=1) / c.float().norm(dim=1).clamp_min(1e-8)
        torch.testing.assert_close(d['relation_to_cls_norm_ratio'],expected.detach())
        output.sum().backward()
        self.assertIsNotNone(r.grad)


if __name__ == '__main__':
    unittest.main()
