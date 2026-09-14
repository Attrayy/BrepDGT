import torch
from torch import nn
import numpy as np
from typing import Union
from brepdgt.models.backbones.dit_blocks import (
    Mlp,
    modulate,
    TimestepEmbedder,
    FinalLayer,
    get_1d_sincos_pos_embed_from_grid,
)


class ResidualBlock(nn.Module):
    """
    This is similar to a DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, except attention has been
    removed.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm1(x), shift_mlp, scale_mlp)
        )
        return x


class ResidualSingleLatent(nn.Module):
    name = "residual_single_latent"

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        z_dim: int,
        z_conditioning: bool,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        use_pe: bool = True,
        final_residual: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        assert self.seq_length == 1
        self.z_dim = z_dim
        self.z_conditioning = z_conditioning

        # DiT specifics
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_pe = use_pe
        assert not self.use_pe
        self.final_residual = final_residual
        self.mlp_ratio = mlp_ratio

        # First project each token to the channel dimension
        self.x_proj = nn.Linear(self.input_dim, self.hidden_size)

        # Will use fixed sin-cos embedding:
        if self.use_pe:
            self.z_pe = nn.Parameter(
                torch.zeros(1, self.seq_length, self.hidden_size),
                requires_grad=False,
            )

        self.time_embedding = TimestepEmbedder(hidden_size)

        if self.z_conditioning:
            self.z_embedding = nn.Sequential(
                nn.Linear(self.z_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size, out_channels=self.input_dim
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.use_pe:
            pos = np.arange(self.seq_length, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.z_pe.shape[-1], pos)
            self.z_pe.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, z: Union[torch.Tensor, None] = None
    ):
        # This one assumes channels at the end...

        # Positional encoding & time embedding
        t_emb = self.time_embedding(t)
        c_emb = t_emb

        # Project to hidden dimension
        h = self.x_proj(x)

        if self.use_pe:
            h += self.z_pe

        # Apply each block
        for block in self.blocks:
            h = block(h, c_emb)

        # Project back and add the residual
        if self.final_residual:
            h = self.final_layer(h, c_emb) + x
        else:
            # assert False
            h = self.final_layer(h, c_emb)

        return h

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,
        cfg_scale: float = 1.0,
    ):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, z)
        model_out = model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
