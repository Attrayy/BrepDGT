from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
from torch import nn


from brepdgt.models.backbones.dit_blocks import (
    BRepDGTBlock,
    DiTBlockCrossAttn,
    FinalLayer,
    TimestepEmbedder,
    ZEmbedder,
    modulate,
    get_1d_sincos_pos_embed_from_grid,
)


class Dit1D(nn.Module):
    name = "dit_1d"

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        z_dim: int,
        n_z: int,
        z_conditioning: bool,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        use_pe: bool = True,
        final_residual: bool = True,
        mlp_ratio: float = 4.0,
        uncond_prob: float = 0.1,
        dropout: float = 0.0,
        t_low_timesteps: int = 500,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.z_dim = z_dim
        self.z_conditioning = z_conditioning
        self.dropout = dropout

        # DiT specifics
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_pe = use_pe
        self.final_residual = final_residual
        self.mlp_ratio = mlp_ratio
        self.n_z = n_z

        # First project each token to the channel dimension
        self.x_proj = nn.Linear(input_dim, self.hidden_size)

        # Will use fixed sin-cos embedding:
        if self.use_pe:
            self.z_pe = nn.Parameter(
                torch.zeros(1, self.seq_length, self.hidden_size),
                requires_grad=False,
            )

        self.time_embedding = TimestepEmbedder(hidden_size)

        if self.z_conditioning:
            self.z_embedding = ZEmbedder(
                z_dim=self.z_dim,
                n_z=self.n_z,
                hidden_size=self.hidden_size,
                uncond_prob=uncond_prob,
                dropout=dropout,
            )

        self.blocks = nn.ModuleList(
            [
                BRepDGTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=dropout,
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

        # Initialize z embedding MLP:
        if self.z_conditioning:
            nn.init.normal_(self.z_embedding.proj.fc1.weight, std=0.02)
            nn.init.normal_(self.z_embedding.proj.fc2.weight, std=0.02)

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
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        This one assumes channels at the end...
        :param x: Tensor of B x n_seq x C
        :param t:
        :param z:
        :param mask:
        :return:
            h: Tensor of B x n_seq x C
        """

        # Positional encoding & time embedding
        t_emb = self.time_embedding(t)

        if self.z_conditioning:
            z_emb = self.z_embedding(z)
            # c_emb = t_emb + z_emb
            c_emb = t_emb
        else:
            c_emb = t_emb

        # Project to hidden dimension
        h = self.x_proj(x)

        if self.use_pe:
            h += self.z_pe
        if self.z_conditioning:
            h += z_emb

        # Apply each block
        for block in self.blocks:
            h = block(h, c_emb, mask=mask)

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
        mask: Union[torch.Tensor, None] = None,
    ):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, z, mask=mask)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


class Dit1DCrossAttn(nn.Module):
    name = "dit_1d_cross_attn"

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        z_dim: int,
        n_z: int,
        z_conditioning: bool,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        use_pe: bool = True,
        final_residual: bool = True,
        mlp_ratio: float = 4.0,
        uncond_prob: float = 0.1,
        dropout: float = 0.0,
        output_dim: Union[int, None] = None,
        t_low_timesteps: int = 500,
    ):
        super().__init__()
        assert z_conditioning

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.z_dim = z_dim
        self.n_z = n_z
        self.z_conditioning = z_conditioning
        self.dropout = dropout

        # DiT specifics
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_pe = use_pe
        self.final_residual = final_residual
        self.mlp_ratio = mlp_ratio

        # First project each token to the channel dimension
        self.x_proj = nn.Linear(input_dim, self.hidden_size)

        # Will use fixed sin-cos embedding:
        if self.use_pe:
            self.z_pe = nn.Parameter(
                torch.zeros(1, self.seq_length, self.hidden_size),
                requires_grad=False,
            )

        self.time_embedding = TimestepEmbedder(hidden_size)
        self.z_embedding = ZEmbedder(
            z_dim=self.z_dim,
            n_z=self.n_z,
            hidden_size=self.hidden_size,
            uncond_prob=uncond_prob,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlockCrossAttn(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size,
            out_channels=self.input_dim if self.output_dim is None else self.output_dim,
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

        # Initialize z embedding MLP:
        if self.z_conditioning:
            nn.init.normal_(self.z_embedding.proj.fc1.weight, std=0.02)
            nn.init.normal_(self.z_embedding.proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out cross-attention layers in DiT blocks:
        # cf. PixArt-alpha (sec 2.3): https://arxiv.org/abs/2310.00426
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.out_proj.weight, 0)
            nn.init.constant_(block.cross_attn.out_proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        This one assumes channels at the end...
        :param x: Tensor of B x n_seq x C
        :param t:
        :param z:
        :param mask:
        :return:
        """

        # Positional encoding & time embedding
        t_emb = self.time_embedding(t)
        z_emb = self.z_embedding(z)

        # Project to hidden dimension
        h = self.x_proj(x)

        if self.use_pe:
            h += self.z_pe

        # Apply each block
        for block in self.blocks:
            h = block(h, c=z_emb, t=t_emb, mask=mask)

        # Project back and add the residual
        if self.final_residual:
            h = self.final_layer(h, c=t_emb) + x
        else:
            # assert False
            h = self.final_layer(h, c=t_emb)

        return h

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,
        cfg_scale: float = 1.0,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        This one assumes channels at the end...
        :param x: Tensor of B x n_seq x C
        :param t:
        :param z:
        :param mask:
        :return:
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, z, mask=mask)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


class TwoDit1D(nn.Module):
    name = "two_dit_1d"

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        z_dim: int,
        n_z: int,
        z_conditioning: bool,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        use_pe: bool = True,
        final_residual: bool = True,
        mlp_ratio: float = 4.0,
        uncond_prob: float = 0.1,
        dropout: float = 0.0,
        t_low_timesteps: int = 500,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.z_dim = z_dim
        self.z_conditioning = z_conditioning
        self.dropout = dropout

        # network for low noise
        self.dit_1d_t_low = Dit1D(
            input_dim=input_dim,
            seq_length=seq_length,
            z_dim=z_dim,
            n_z=n_z,
            z_conditioning=z_conditioning,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_pe=use_pe,
            final_residual=final_residual,
            mlp_ratio=mlp_ratio,
            uncond_prob=uncond_prob,
            dropout=dropout,
        )

        # network for high noise, coordinates only
        high_t_input_dim = int(input_dim * 3 / 4)  # remove grid_mask
        self.dit_1d_t_high = Dit1D(
            input_dim=high_t_input_dim,
            seq_length=seq_length,
            z_dim=z_dim,
            n_z=n_z,
            z_conditioning=z_conditioning,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_pe=use_pe,
            final_residual=final_residual,
            mlp_ratio=mlp_ratio,
            uncond_prob=uncond_prob,
            dropout=dropout,
        )

        self.t_low_timesteps = t_low_timesteps

        # assert (
        #     input_dim == 256
        # ), f"Currently hard coded for augmented uvgrid with dim 256, got {input_dim}"
        # Precompute where coordinates lies in x
        x_dim_arange = torch.arange(input_dim)
        per_points_dim = 4  # coord (3) + grid (1)
        self.x_coord_mask = (
            (x_dim_arange % per_points_dim == 0)
            | (x_dim_arange % per_points_dim == 1)
            | (x_dim_arange % per_points_dim == 2)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,  # should be None
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        This one assumes channels at the end...
        :param x: Tensor of B x n_seq x C
        :param t: Tensor of B
        :param z:
        :param mask:
        :return:
        """

        t_low_mask = t < self.t_low_timesteps
        # low noise network
        if t_low_mask.sum() != 0:
            out_t_low = self.dit_1d_t_low(
                x=x[t_low_mask],
                t=t[t_low_mask],
                z=None,
                mask=mask[t_low_mask],
            )
        else:
            out_t_low = x[0:0]

        # High t = high noise
        # Only coordinates gets in:
        # `coord -> coord`
        if (~t_low_mask).sum() != 0:
            x_t_high = x[~t_low_mask]

            out_t_high_coords = self.dit_1d_t_high(
                x=x_t_high[:, :, self.x_coord_mask],
                t=t[~t_low_mask],
                z=None,
                mask=mask[~t_low_mask],
            )

            # This is the finicky part, you can keep identity
            # because noise will remain noise for this one
            # It's like the reverse SDE with the identity denoiser :D
            out_t_high = torch.zeros_like(x_t_high)
            out_t_high[:, :, self.x_coord_mask] = out_t_high_coords
            out_t_high[:, :, ~self.x_coord_mask] = x_t_high[:, :, ~self.x_coord_mask]
        else:
            out_t_high = x[0:0]

        out = torch.zeros_like(x)
        out[t_low_mask] = out_t_low
        out[~t_low_mask] = out_t_high

        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: Union[torch.Tensor, None] = None,
        cfg_scale: float = 1.0,
        mask: Union[torch.Tensor, None] = None,
    ):
        raise NotImplementedError("CFG without cross attention should not be called")
