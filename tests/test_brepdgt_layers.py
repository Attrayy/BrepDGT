import unittest

import torch
from torch import nn

from brepdgt.models.layers import DynamicTanh, SCEA
from brepdgt.models.backbones.dit_blocks import BRepDGTBlock


class BRepDGTLayerTests(unittest.TestCase):
    def test_dynamic_tanh_shape_and_initial_state(self):
        layer = DynamicTanh(16)
        x = torch.randn(2, 5, 16)
        y = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(layer.alpha.shape, torch.Size([]))
        self.assertTrue(torch.allclose(layer.gamma, torch.ones_like(layer.gamma)))
        self.assertTrue(torch.allclose(layer.beta, torch.zeros_like(layer.beta)))

    def test_scea_shape_and_checkpoint_keys(self):
        layer = SCEA(dim=48, num_heads=6, kv_heads=3)
        y = layer(torch.randn(2, 7, 48))

        self.assertEqual(y.shape, (2, 7, 48))
        state = layer.state_dict()
        self.assertEqual(state["q_proj.weight"].shape, (48, 48))
        self.assertEqual(state["k_proj.weight"].shape, (24, 48))
        self.assertEqual(state["v_proj.weight"].shape, (24, 48))
        self.assertEqual(state["w_post"].shape, (6, 6))

    def test_scea_ignores_masked_keys(self):
        torch.manual_seed(7)
        layer = SCEA(dim=32, num_heads=4, kv_heads=2).eval()
        x = torch.randn(1, 6, 32)
        mask = torch.tensor([[False, False, False, False, False, True]])

        changed = x.clone()
        changed[:, -1] = 1000.0
        actual = layer(x, key_padding_mask=mask)
        expected = layer(changed, key_padding_mask=mask)

        torch.testing.assert_close(actual[:, :-1], expected[:, :-1])

    def test_scea_uses_fewer_projection_parameters_than_mha(self):
        scea = SCEA(dim=768, num_heads=12, kv_heads=6)
        mha = nn.MultiheadAttention(768, 12, batch_first=True)

        scea_params = sum(parameter.numel() for parameter in scea.parameters())
        mha_params = sum(parameter.numel() for parameter in mha.parameters())
        self.assertLess(scea_params, mha_params)

    def test_main_block_uses_scea_and_dyt(self):
        block = BRepDGTBlock(hidden_size=48, num_heads=6)
        self.assertIsInstance(block.attn, SCEA)
        self.assertIsInstance(block.norm1, DynamicTanh)
        self.assertIsInstance(block.norm2, DynamicTanh)

        x = torch.randn(2, 7, 48)
        condition = torch.randn(2, 48)
        self.assertEqual(block(x, condition).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
