import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import Tensor
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

'''
img = Image.open('D:/MyViT/data/0001.png')
fig = plt.figure()
plt.imshow(img)
# plt.show()

# 预处理图片
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
x = transform(img)
x = x.unsqueeze(0)
print(x.shape)

# 将图片分成小patch
patch_size = 16


# patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
# print(patches.shape)
'''

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel, emb_size, kernel_size=patch_size, stride=patch_size),  # 通过768个卷积核使每个patch有768个值表示
            Rearrange('b e h w -> b (h w) e'),  # 重新指定维度,(b,14*14个token,768个h_dim)
            # nn.Linear(patch_size * patch_size * in_channel, emb_size)

        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码信息，随机初始化，然后让模型自己去学习
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # 不理会batch的合并，196块patch加1个cls token=197个维度来描述输入图像信息
        x += self.positions  # 每个batch上都加了一个位置编码
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)  # 将输入a1234...映射为qkv1234...,一个patch是16*16*3个像素信息
        self.att_drop = nn.Dropout(dropout)  # 以dropout的概率将元素置为0
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = self.qkv(x)  # QKV矩阵，x:(b,n,16*16*3)->qkv:(b,n,16*16*3*3)->(b,n,8*96*3)
        qkv = rearrange(qkv, 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads,  # (1，197，8*96*3)->(3,1,8,197,96)
                        qkv=3)  # qkv在前面便于下一步分开它们，n为信息编码长度,d应该是2304分成qkv三份，随后分成head8份后的剩余数而已
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b,h,n,d) h在前因为h是仅次于b的不相关计算划分
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)  # 爱因斯坦求和约定表示的q*kT
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min  # 一个非常大的float32型负数
            energy.mask_fill(~mask, fill_value)  # ?
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling  # todo
        # att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)  # ?
        out = torch.einsum('bhal, bhlv -> bhav', att, v)  # att*v得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)  # FC
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)  # 应该就是残差结构跳过的函数
        x += res
        return x


class FeedForwardBlock(nn.Sequential):  # 继承sequential类，可以避免重写forward方法
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


# 完整的transformer encoder块
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


# 最终分类的全连接层
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_class=1000):
        super(ClassificationHead, self).__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_class)
        )


# 将前面的class整合为一个ViT
class ViT(nn.Sequential):
    def __init__(self, in_channel=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_class=1000, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channel, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_class)
        )

# patches = PatchEmbedding()
# result = patches(x)  # 将输入信息embedding
# print(result.shape)
# # multiheadatt = MultiHeadAttention()
# # out_ = multiheadatt(result)
# # print(out_.shape)
# encoder = TransformerEncoderBlock()
# out = encoder(result)
# print(out.shape)

# myvit = ViT()
# summary(myvit.cuda(), (3, 224, 224))
# result = myvit(x)
# print(result.shape)
