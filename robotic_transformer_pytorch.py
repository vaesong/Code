import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange
import einops
from pathlib import Path
import matplotlib.pyplot as plt

from typing import List, Optional, Callable, Tuple, Union
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from functools import partial

from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)
# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, cond_fn = None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
        residual=False,
    ):
        super().__init__()
        self._residual = residual

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)
        res = out.clone()

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)
            res = self.activation(res)

        if self._residual:
            return out, res
        else:
            return out

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem
        self.channels = channels

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        x = self.conv_stem(x)

        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers)

        for stage, cond_fn in zip(self.layers, cond_fns):
            # 使用 text embedding 指导 image embedding
            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)    # torch.Size([48, 64, 64, 64])

        if return_embeddings:
            return x

        return self.mlp_head(x)

# attention

class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers) * 2

        cond_fns = iter(cond_fns)

        for attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns)) + x
             x = ff(x, cond_fn = next(cond_fns)) + x
        return x

# CSLearner
class CSALearner(nn.Module):
    def __init__(
        self,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        ratio=16,        
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc = nn.Sequential(
            nn.Linear(dim, dim//ratio, False),
            nn.ReLU(),
            nn.Linear(dim//ratio, dim, False),
            nn.Sigmoid()
        )
        
        inner_dim = dim * ff_mult * num_output_tokens
        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )
        
    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        # b, t, n, c, h, w= x.shape
        # x = rearrange('b, t, n, c, h, w -> (b t) (n c) h w')
        
        b, c, _, _= x.size() #取出batch size和通道数
        
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg = self.avgpool(x).view(b, c)
        #avg = reduce(x, 'b c w h -> b c', 'mean')
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc = self.fc(avg).view(b, c, 1, 1)
        
        x = fc * x
        
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x
        
        
# SEnet
class SEnet(nn.Module):
    def __init__(
        self,
        channels,
        ratio=16
    ):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1) 

        return fc*x

# token learner module
class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x

# Robotic Transformer

@beartype
class RT1(nn.Module):
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions = 11,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 16,
        cond_drop_prob = 0.2,
        dropout = 0.1,
        cnn_depth = 4,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        instr_size: int = 768,
        num_tasks: int = 10,
        maxAction: int = 6,
        num_cams: int = 3,
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.dropout = dropout
        self.vit = vit

        self.num_vit_stages = len(vit.cond_hidden_dims)

        conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

        self.conditioner = conditioner_klass(
            hidden_dims = (*tuple(vit.cond_hidden_dims), *((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
            cond_drop_prob = cond_drop_prob,
            **conditioner_kwargs
        )

        # self.token_learner = TokenLearner(
        #     dim = vit.embed_dim,
        #     ff_mult = token_learner_ff_mult,
        #     num_output_tokens = token_learner_num_output_tokens,
        #     num_layers = token_learner_num_layers
        # )
        
        self.csa_learner = CSALearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
        )
        # self.cam_embedding = nn.Embedding(num_cams, vit.embed_dim)
        # self.cam_norm = nn.LayerNorm(vit.embed_dim)

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth
        self.cnn_depth = cnn_depth

        self.rgb_preprocess = ConvLayer(
            self.vit.channels,
            8,
            kernel_size=(3, 3),
            stride_size=(1, 1),
            apply_norm=False,
        )
        self.to_feat = ConvLayer(
            8,
            16,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
        )

        # Encoder-Decoder Network, maps to pixel location with spatial argmax
        self.feature_encoder = nn.ModuleList()
        for i in range(self.cnn_depth):
            self.feature_encoder.append(
                ConvLayer(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    stride_size=(2, 2),
                    residual=True,
                )
            )

        self.trans_decoder = nn.ModuleList()
        for i in range(self.cnn_depth):
            if i == 0:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=6,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
            elif i == self.cnn_depth - 1:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=16 * 2,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
            else:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=16 * 2,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )

        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth,
            attn_dropout=dropout,
            ff_dropout=dropout,
        )

        # self.to_logits = nn.Sequential(
        #     LayerNorm(vit.embed_dim), # (B, T, 2048)
        #     nn.Linear(vit.embed_dim, num_actions * action_bins),  # (B, T, 8*256)
        #     Rearrange('... (a b) -> ... a b', b = action_bins)    # (B, T, 8, 256)
        # )

        self.maps_to_coord = ConvLayer(
            in_channels=16,
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )

        # self.quat_decoder = nn.Sequential(
        #     ConvLayer(
        #         in_channels=quat_hidden_size, out_channels=quat_hidden_size,
        #         kernel_size=(3, 3), stride_size=(2, 2),
        #     ),
        #     ConvLayer(
        #         in_channels=quat_hidden_size, out_channels=quat_hidden_size,
        #         kernel_size=(3, 3), stride_size=(2, 2)
        #     ),
        #     nn.AdaptiveAvgPool2d(1),
        #     Rearrange("b c h w -> b (c h w)"),
        #     *dense_layer(quat_hidden_size, quat_hidden_size),
        #     *dense_layer(quat_hidden_size, 3 + 4 + 1, apply_activation=False),
        # )
        
        # self.z_proj_instr = nn.Linear(instr_size, num_tasks)
        # self.z_pos_instr = nn.Embedding(maxAction, 3 * num_tasks)
        # self.z_pos_instr.weight.data.fill_(0)  # type: ignore
        # self.z_proj_instr.weight.data.fill_(0)  # type: ignore
        # self.z_proj_instr.bias.data.fill_(0)  # type: ignore

        self.predict_action = nn.Sequential(
            LayerNorm(vit.embed_dim * 3),
            nn.Linear(vit.embed_dim * 3, vit.embed_dim),
            nn.ReLU(),
            nn.Linear(vit.embed_dim, num_actions),
            nn.Tanh(),
        )

    def head(
        self, 
        x : torch.Tensor,
        rgb_obs: torch.Tensor,
        pc_obs: torch.Tensor,
        enc_feat: List,
        padding_mask: torch.Tensor,
        instruction: torch.Tensor,
    ):
        B = pc_obs.shape[0]
        T = pc_obs.shape[1]
        N = pc_obs.shape[2]
        device = pc_obs.device

        pc_obs = pc_obs[padding_mask]
        rgb_obs = rgb_obs[padding_mask]
        enc_feat.reverse()        

        x = einops.rearrange(x, "b t n (c h w) -> b t n c h w", h=8, w=8)
        x = x[padding_mask]
        x = einops.rearrange(x, "bpad n c h w -> (bpad n) c h w")

        for i, l in enumerate(self.trans_decoder):
            if i == 0:
                xpr = self.trans_decoder[0](x)
            else:
                xpr = l(torch.cat([xpr, enc_feat[i]], dim=1))

        xp = xpr

        xp = self.maps_to_coord(xp)
        xp = einops.rearrange(xp, "(bpad n) c h w -> bpad (n c h w)", n=N, c=1)
        xp = torch.softmax(xp / 0.1, dim=1)
        # attn_map = einops.rearrange(
        #     xp, "bpad (n c h w) -> bpad n c h w", n=N, c=1, h=128, w=128
        # )

        attn_map = einops.rearrange(
            xp, "bpad (n c h w) -> bpad n c h w", n=N, c=1, h=128, w=128
        )
        
        # plt.imshow(attn_map.squeeze(0).permute(1, 2, 0).detach().cpu(), cmap='Greys_r')
        # plt.axis("off")
        # plt.savefig(Path('/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/xp/masks') / f"mask", bbox_inches="tight", pad_inches = -0.1)
        # plt.clf()

        # ------------------ test --------------------------------
        # attn_map = einops.rearrange(attn_map, 'bpad c h w -> bpad (h w) c')
        # pcds = einops.rearrange(pc_obs, 'bpad c h w -> bpad (h w) c')
        # selected_points = pcds * attn_map

        # position = torch.Tensor([]).to(device)
        # for i in range(B):
        #     attn = einops.rearrange(attn_map[i], 'c h w -> (h w) c')
        #     mask = torch.zeros_like(attn)
        #     sorted_attn, indices = torch.sort(attn[:,0], descending=True)

        #     # mask[indices[:20]] = 1
        #     # mask = einops.rearrange(mask, '(h w) c -> h w c', h=128, w=128)
        #     # plt.imshow(mask.detach().cpu(), cmap='Greys_r')
        #     # plt.axis("off")
        #     # plt.savefig(Path('/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/xp/masks') / f"mask1", bbox_inches="tight", pad_inches = -0.1)
        #     # plt.clf()

        #     pcd = einops.rearrange(pc_obs[i], 'c h w -> (h w) c')
        #     sorted_pcd = pcd[indices]
        #     top = sorted_pcd[:20]
        #     x_mean = torch.mean(top[:, :1])
        #     y_mean = torch.mean(top[:, 1:2])
        #     z_mean = torch.mean(top[:, 2:])
            
        #     pos = torch.tensor([x_mean, y_mean, z_mean])

        #     position = torch.cat([position, pos.unsqueeze(0).to(device)], dim=0)

        # plt.imshow(attn_map.squeeze(0).permute(1, 2, 0).detach().cpu(), cmap='Greys_r')
        # plt.axis("off")
        # plt.savefig(Path('/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/xp/masks') / f"mask", bbox_inches="tight", pad_inches = -0.1)
        # plt.clf()

        # rgb = rgb_obs.squeeze(0).detach().cpu()
        # plt.imshow(rgb.permute(1, 2, 0))
        # plt.axis("off")
        # plt.savefig(Path('/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/xp/masks') / f"rgb", bbox_inches="tight", pad_inches = -0.1)
        # plt.clf()

        position = einops.reduce(pc_obs * attn_map, "bpad n c h w -> bpad c", "sum")
        
        # # # prediction offset position
        # # g = instruction
        # task = self.z_proj_instr(instruction)
        # num_tasks = task.shape[1]
        # z_instr = task.softmax(1)
        # z_instr = einops.repeat(z_instr, "b n -> b t 1 n", t=T)
        # z_instr = z_instr[padding_mask]

        # step_ids = torch.arange(T, dtype=torch.long, device=device)
        # z_pos = self.z_pos_instr(step_ids.unsqueeze(0)).squeeze(0)
        # z_pos = einops.repeat(z_pos, "t (n d) -> b t n d", b=B, n=num_tasks, d=3)
        # z_pos = z_pos[padding_mask]

        # z_offset = torch.bmm(z_instr, z_pos).squeeze(1)
        # position += z_offset

        xr = einops.rearrange(x, "(bpad n) c h w -> bpad (n c h w)", n=N)
        xr = self.predict_action(xr)
        # xr = xr[padding_mask]
        rotation = xr[:, 3:7]
        rotation = normalise_quat(rotation)
        gripper = torch.sigmoid(xr[:, -1:])

        action = torch.cat([position, rotation, gripper], dim=1)
        # return action
        return {
            "action": action,
            "attention": attn_map
        }



    @classifier_free_guidance
    def forward(
        self,
        video,
        texts,
        pc_obs,
        padding_mask,
        cond_drop_prob = 0.
    ):
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        B = video.shape[0]
        T = video.shape[1]
        N = video.shape[2]
        # N = 1

        frames, device = video.shape[1], video.device

        cond_fns = self.conditioner(
            texts,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames * N,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        # video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')   # torch.Size([3, 3, 128, 128])

        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )
        # torch.Size([3, 384, 8, 8])
        tokens = unpack_one(tokens, packed_shape, '* c h w')    # B * T * D * h * w -> B, T, N, C, h, w
        # print(tokens.shape, "===============")
        
        # learned_tokens = self.token_learner(tokens) # B * T * D * s -> B, T, N, D, s
        # learned_tokens = rearrange(tokens, 'b t n c h w -> b t n c (h w)')
        learned_tokens = self.csa_learner(tokens)

        # learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        learned_tokens = rearrange(learned_tokens, 'b t n c s -> b (t n s) c')

        # causal attention mask

        attn_mask = torch.ones((T, T), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens * N, r2 = self.num_learned_tokens * N) # (f n), (f n) -> 

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(T, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        # cam_id = torch.arange(N).type_as(learned_tokens).unsqueeze(0).long()
        # cam_emb = self.cam_embedding(cam_id)
        # cam_emb = self.cam_norm(cam_emb).squeeze(0)
        # cam_emb = einops.repeat(cam_emb, "n d -> (t n s) d", s = self.num_learned_tokens, t = T)
        
        learned_tokens = learned_tokens + repeat(pos_emb, 't d -> (t n s) d', s = self.num_learned_tokens, n = N)

        # ****************************** 提取图像特征 ****************************************
        video_ = einops.rearrange(video, "b t n ch h w -> (b t n) ch h w")

        video_ = self.rgb_preprocess(video_)

        x = self.to_feat(video_)
        # encoding features
        enc_feat = []
        for l in self.feature_encoder:
            x, res = l(x)

            res = einops.rearrange(res, "(b t n) c h w -> b t n c h w", n=N, t=T)  
            res = res[padding_mask]
            res = einops.rearrange(res, "bpad n c h w -> (bpad n) c h w")
            enc_feat.append(res)
        # ****************************** 提取图像特征 ****************************************

        # attention
        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = ~attn_mask)

        # pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)
        pooled = reduce(attended_tokens, 'b (t n l) c -> b t n c', 'mean', t = T, n = N)


        # torch.Size([8, 6, 384]) ----------> torch.Size([8, 6, 384])
        # torch.Size([8, 6, 384]) ----------> torch.Size([8, 6, 2048])
        # torch.Size([8, 6, 2048]) ----------> torch.Size([8, 6, 8, 256])
        # print("********************************")

        # pooled = einops.rearrange(pooled, 'b t n c -> b t (n c)')
        # pooled = pooled[padding_mask]
        # logits = self.predict_action(pooled)
        # logits[:, 3:7] = normalise_quat(logits[:, 3:7])
        # return {
        #     "action": logits,
        #     "attention": None
        # }

        text_embed = self.conditioner.embed_texts(texts=texts)
        logits = self.head(pooled, video, pc_obs, enc_feat, padding_mask, text_embed)
        return logits

        
