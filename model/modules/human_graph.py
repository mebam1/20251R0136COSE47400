import torch
from itertools import product
from torch.linalg import eigh

class CachedGraph():
    @torch.no_grad()
    def __init__(self, edge_index:torch.Tensor, num_nodes:int, mode:str):
        if mode == 'skeleton':
            one_hop_adj = CachedGraph.get_adj(num_nodes, edge_index)
            adj = one_hop_adj
            adj.fill_diagonal_(1)
            self.edge_index = self.get_edge_index(adj)
            self.shortest_distance = floyd_warshall(adj)
            self.num_nodes = num_nodes

        elif mode == 'cayley':
            self.edge_index = edge_index
            self.num_nodes = num_nodes
        else:
            raise NotImplementedError()


    @staticmethod
    @torch.no_grad()
    def get_adj(num_nodes:int, edge_index:torch.Tensor) -> torch.Tensor:
        adj = edge_index.new_zeros((num_nodes, num_nodes), dtype=torch.long)
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1
        return adj
    
    @staticmethod
    @torch.no_grad()
    def get_edge_index(adj:torch.Tensor) -> torch.Tensor:
        src, dst = adj.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)

        
        new_edges = [
            (17, 16), (17, 18),
            (18, 13),
            (19, 3),  (19, 20),
            (20, 6),
            (21, 14), (21, 15), (21, 22),
            (22, 11), (22, 12),
            (23, 2), (23, 5)
        ]

        bidirectional_edges = []
        for u, v in new_edges:
            bidirectional_edges.append((u, v))
            bidirectional_edges.append((v, u))

        for u in range(17, 24):
            bidirectional_edges.append((u, u))

        new_edge_index = torch.tensor(bidirectional_edges, dtype=torch.long, device=edge_index.device).t()  # shape: [2, num_new_edges]
        e = torch.cat([edge_index, new_edge_index], dim=-1)
        return e


    @staticmethod
    @torch.no_grad()
    def build_human_graph(mode:str='skeleton'):

        device ='cuda' if torch.cuda.is_available() else 'cpu'

        if mode == 'skeleton':
            x = torch.tensor([
                [
                    0,0,0,1,1,2,2,3,4,4,5,5,6,7,7,8,8,8,8,9,9,10,11,11,12,12,13,14,14,15,15,16
                ],
                [
                    1,4,7,0,2,1,3,2,0,5,4,6,5,0,8,7,9,11,14,8,10,9,8,12,11,13,12,8,15,14,16,15
                ]
                ], device=device, dtype=torch.long)
            return CachedGraph(x, 17, mode)
        elif mode == 'cayley':
            x, n = Cayley(device)
            return CachedGraph(x, n, mode)

def det_mod3(a, b, c, d):
    return (a * d - b * c) % 3 == 1

def matmul_mod3(m1, m2):
    a1, b1 = m1[0]
    c1, d1 = m1[1]
    a2, b2 = m2[0]
    c2, d2 = m2[1]
    return (
        (( (a1 * a2 + b1 * c2) % 3, (a1 * b2 + b1 * d2) % 3 ),
         ( (c1 * a2 + d1 * c2) % 3, (c1 * b2 + d1 * d2) % 3 ))
    )

@torch.no_grad()
def Cayley(device):
    Z3 = [0, 1, 2]
    SL2_Z3 = []

    for a, b, c, d in product(Z3, repeat=4):
        if det_mod3(a, b, c, d):
            SL2_Z3.append(((a, b), (c, d)))

    mat2idx = {m: i for i, m in enumerate(SL2_Z3)}

    S3 = [
        ((1, 1), (0, 1)),
        ((1, 0), (1, 1)),
    ]

    edges = []

    for g in SL2_Z3:
        u = mat2idx[g]
        for s in S3:
            gs = matmul_mod3(g, s)
            if gs in mat2idx:
                v = mat2idx[gs]
                edges.append((u, v))
                edges.append((v, u))  # 무방향 edge

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index, len(SL2_Z3)

@torch.no_grad()
def GetPosEnc(adj, device):
    # D: degree matrix, adj: [N, N] adjacency matrix
    N = adj.shape[0]
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.diag(torch.pow(deg.clamp(min=1e-8), -0.5)) # D^{-1/2}
    I = torch.eye(N, device=device)
    L = I - deg_inv_sqrt @ adj @ deg_inv_sqrt
    
    eigval, eigvec = eigh(L) # eigen-decomposition
    sorted_index = torch.argsort(eigval)
    eigvec = eigvec[:, sorted_index[1:]] # [N, 16] positional encoding
    print(eigval)
    return eigvec, eigval, L

@torch.no_grad()
def floyd_warshall(adj:torch.Tensor, inf:float=1e9):
    A = adj.clone()
    N = A.shape[0]

    mask_no_edge = (A == 0)
    A[mask_no_edge] = inf
    # 무가중치(0/1) 입력이면 1로
    #A[(mask_no_edge == False) & (A == 1)] = 1
    
    D = A.clone()        # 거리 행렬
    
    # 2단계 ─ Floyd–Warshall O(N³)
    for k in range(N):
        # broadcasting: D[:, k][:, None] + D[k, :]
        D = torch.minimum(D, D[:, k][:, None] + D[k, :])

    D.fill_diagonal_(0)
    return D

if __name__ == '__main__':
    g = CachedGraph.build_human_graph()