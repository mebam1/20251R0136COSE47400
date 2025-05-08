import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAT
import torch_geometric.data as graph
from model.modules.mlp import MLP

class BatchedCachedGraph():
    def __init__(self, graph_data: graph.Data):
        self.batched_edge_index = None
        self.num_nodes = graph_data.num_nodes
        self.edge_index = graph_data.edge_index

    @torch.no_grad()
    def get_batched_edge_index(self, batch_size:int):
        if (self.batched_edge_index is not None) and (self.batched_edge_index.shape[1] == self.edge_index.shape[1] * batch_size):
            return self.batched_edge_index
        
        E = self.edge_index.shape[1]
        offsets = torch.arange(batch_size, device=self.edge_index.device) * self.num_nodes
        offsets = offsets.repeat_interleave(E)
        self.batched_edge_index = self.edge_index.repeat(1, batch_size) + offsets
        return self.batched_edge_index

    @staticmethod
    @torch.no_grad()
    def build_human_graph(joint_or_bone:str):
        if joint_or_bone not in ['joint', 'bone']:
            raise ValueError("joint_or_bone must be 'joint' or 'bone'")
        device ='cuda' if torch.cuda.is_available() else 'cpu'

        if joint_or_bone == 'joint':
            h36m_joint_edge_tensor = torch.tensor([
                [
                    0,0,0,1,1,2,2,3,4,4,5,5,6,7,7,8,8,8,8,9,9,10,11,11,12,12,13,14,14,15,15,16
                ],
                [
                    1,4,7,0,2,1,3,2,0,5,4,6,5,0,8,7,9,11,14,8,10,9,8,12,11,13,12,8,15,14,16,15
                ]
                ], device=device)
            h36m_joint_graph = graph.Data(edge_index=torch.cat([h36m_joint_edge_tensor, h36m_joint_edge_tensor.flip(0)],dim=1))
            h36m_joint_graph.num_nodes=17
            return BatchedCachedGraph(h36m_joint_graph)
        else:
            bone_edge_tensor = torch.tensor([
                [
                    0,0,0,1,1,2,3,3,3,4,4,5,6,6,6,7,7,7,7,8,8,8,8,9,10,10,10,10,11,11,12,13,13,13,13,14,14,15
                ],
                [
                    1,3,6,0,2,1,0,4,6,3,5,4,0,3,7,6,8,10,13,7,9,10,13,8,7,8,11,13,10,12,11,7,8,10,14,13,15,14
                ]
                ], device=device)
            bone_graph = graph.Data(edge_index=torch.cat([bone_edge_tensor, bone_edge_tensor.flip(0)],dim=1))
            bone_graph.num_nodes=16
            return BatchedCachedGraph(bone_graph)


j_graph=BatchedCachedGraph.build_human_graph('joint')
#b_graph=BatchedCachedGraph.build_human_graph('bone')

class RGAT(nn.Module):
    def __init__(self, dim, j_or_e, use_GATv2:bool=True, appnp_iter:int=1, drop:float=0.0):
        super().__init__()
        self.human_graph = j_or_e
        self.appnp1 = GATAppnp(dim, use_GATv2)
        #self.norm = nn.LayerNorm(dim)
        #self.mlp = MLP(dim, 4*dim, dim, drop=drop)

    def forward(self, x: torch.Tensor):
        B,T,J,C = x.shape
        x = x.view(-1, C)
        x2 = self.appnp1(x, self.human_graph)
        x = x + x2
        x = x.view(B,T,J,C)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_dim:int, out_dim:int|None, use_GATv2:bool=True, appnp_iter:int=1, drop:float=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.spatial = nn.Sequential(
            RGAT(in_dim, j_graph, drop=drop),
            nn.LayerNorm(in_dim),
            MLP(in_dim, out_dim * 2, out_dim),
            RGAT(out_dim, j_graph, drop=drop)
        )

    def forward(self, x: torch.Tensor):
        # Consider x.shape: [B,T,J,C].
        x = self.spatial(x)
        return x

class GATAppnp(nn.Module):

    def __init__(self, dim:int, use_GATv2:bool):
        super().__init__()
        self.conv = GAT(dim, dim, num_layers=1, dropout=0.25, act='gelu', v2=use_GATv2, concat=False, heads=1)
        self.alpha=nn.Parameter(torch.tensor(-0.1))  #nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x:torch.Tensor, batched_graph:BatchedCachedGraph):
        # B * T are spatial batch dimension.
        # Consider x.shape: [B*T*J, C].
        edges = batched_graph.get_batched_edge_index(x.shape[0] // batched_graph.num_nodes)
        x0 = x
        x = self.norm(x)
        a = self.alpha
        skip_rate = F.sigmoid(a)

        x = self.conv(x, edges)
        for _ in range(3):
            x = (1.0 - skip_rate) * self.conv(x, edges) + skip_rate * x0
        return x



#==================#
#  UNIT TEST CODE  #
#==================#

def test_spatial_graph_attention():
    # Create sample data
    edge_index = torch.tensor([[0,1], [1,2]], dtype=torch.long)
    
    # Initialize model
    B, T, J, C = 8, 81, 17, 32
    model = RGAT(
        dim=C,
        joint_or_bone='joint',
        use_GATv2=True,
        appnp_iter=8
    )
    
    # Test forward pass
    x = torch.randn(B, T, J, C)
    output = model(x)
    
    assert not torch.allclose(output, x), "Model not modifying input"

if __name__ == '__main__':
    test_spatial_graph_attention()
