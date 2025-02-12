# ---------------------------------------------------------------
# Copyright (c) 2024, NoteDance. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import tensorflow as tf
from Note import nn
from functools import partial
import math
import collections.abc
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)

class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)

    def __call__(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.dense(dim, dim, use_bias=qkv_bias)
        self.kv = nn.dense(dim * 2, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.conv2d(dim, kernel_size=sr_ratio, input_size=dim, strides=sr_ratio)
            self.norm = nn.layer_norm(dim)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        q = tf.transpose(tf.reshape(self.q(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = tf.reshape(x, (B, H, W, C))
            x_ = tf.transpose(tf.reshape(self.sr(x_), (B, C, -1)), (0, 2, 1))
            x_ = self.norm(x_)
            kv = tf.transpose(tf.reshape(self.kv(x_), (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        else:
            kv = tf.transpose(tf.reshape(self.kv(x), (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block:

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, sr_ratio=1):
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed:
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, name=None):
        nn.Model.namespace(name)
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.conv2d(embed_dim, kernel_size=patch_size, input_size=in_chans, strides=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.layer_norm(embed_dim)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H*W, C))
        x = self.norm(x)

        return x, H, W


class MiT(nn.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.layer_norm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        nn.Model.add()
        
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], name='patch_embed1')
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = tf.linspace(0., drop_path_rate, sum(depths))  # stochastic depth decay rule
        cur = 0
        self.block1 = [Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])]
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = [Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])]
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = [Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])]
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = [Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])]
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = self.dense(num_classes, embed_dims[3]) if num_classes > 0 else nn.identity()

        nn.Model.apply(self.init_weights)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.trunc_normal_(l.weight, std=0.2))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            l.weight.assign(nn.initializer_(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def reset_drop_path(self, drop_path_rate):
        dpr = tf.linspace(0., drop_path_rate, sum(self.depths))
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_path_rate = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_path_rate = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_path_rate = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_path_rate = dpr[cur + i]

    def freeze_patch_emb(self):
        self.freeze('patch_embed1')

    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.dense(num_classes, self.embed_dim) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, -1))
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = tf.reshape(x, (B, H, W, -1))
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = tf.reshape(x, (B, H, W, -1))
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = tf.reshape(x, (B, H, W, -1))
        outs.append(x)

        return outs

    def __call__(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv:
    def __init__(self, dim=768):
        self.dwconv = nn.conv2d(dim, 3, dim, 1, groups=dim, padding=1, use_bias=True)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        x = tf.reshape(x, (B, H, W, C))
        x = self.dwconv(x)
        x = tf.reshape(x, (B, H*W, C))

        return x



def mit_b0():
    return MiT(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def mit_b1():
    return MiT(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def mit_b2():
    return MiT(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def mit_b3():
    return MiT(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def mit_b4():
    return MiT(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def mit_b5():
    return MiT(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
