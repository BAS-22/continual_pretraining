
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from models.patch_embed import PatchEmbed
from util.pos_embed import get_3d_sincos_pos_embed


class ViT(nn.Module):
    """VisionTransformer backbones"""

    def __init__(
        self,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def get_num_layers(self):
        return len(self.blocks)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ret_hids=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if ret_hids:
            hidden_states_out = []

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x).contiguous()
            if ret_hids:
                hidden_states_out.append(x[:, 1:, :])
        x = self.norm(x)

        if ret_hids:
            return x[:, 0], x[:, 1:, :], hidden_states_out
        else:
            return x[:, 0], x[:, 1:, :]


def vit_base_patch16_dec512d8b(in_chans=3, dropout_rate=0.1, **kwargs):
    model = ViT(
        img_size=(64, 256, 256),
        patch_size=(16, 32, 32),
        in_chans=in_chans,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

        # Modify the Block instantiations to include dropout
    for i, block in enumerate(model.blocks):
        # Replace existing block with one that has dropout
        model.blocks[i] = Block(
            dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            proj_drop=dropout_rate,
            attn_drop=dropout_rate,
            drop_path=dropout_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    return model


def vit_large_patch16_dec512d8b(**kwargs):
    model = ViT(
        img_size=(64, 256, 256),
        patch_size=(16, 32, 32),
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
vit_base_patch16 = vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vit_large_patch16 = vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


# encoder = vit_base_patch16(in_chans=3)
# msg = encoder.load_state_dict(torch.load(path,map_location='cpu')['model'], strict=False)
# print(msg.missing_keys)
