import torch
from torch import nn
from model.modules.human_graph import CachedGraph


g = CachedGraph.build_human_graph()


class BiasAttention(nn.Module):
    

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='spatial',vis = 'no'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.vis = vis
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # g.shortest_distance: [17, 17] tensor.  
        self.sd_emb = nn.Embedding(9, num_heads)


    def forward(self, x):
        B, T, J, C = x.shape
        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_spatial(q, k, v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn_bias = self.sd_emb(g.shortest_distance[None, None, ...]).permute(0, 4, 1, 2, 3)

        attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)
    

def _test():
    dim=128
    device = 'cuda'
    x = torch.randn(2, 27, 17, dim, device=device)
    model = BiasAttention(dim, dim)
    model.to(device=device)
    y = model(x)


if __name__ == '__main__':
    _test()
