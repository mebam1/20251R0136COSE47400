import torch
from itertools import product
from torch.linalg import eigh
import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci

class CachedGraph():
    @torch.no_grad()
    def __init__(self, edge_index:torch.Tensor, num_nodes:int, mode:str):

        if mode == 'skeleton':
            one_hop_adj = CachedGraph.get_adj(num_nodes, edge_index)
            two_hop_adj = one_hop_adj @ one_hop_adj
            adj = one_hop_adj + two_hop_adj
            adj.fill_diagonal_(0)
            self.edge_index = self.get_edge_index(adj)
            #self.encoding = GetPosEnc(adj, one_hop_adj.device)
            self.num_nodes = num_nodes

        elif mode == 'cayley':
            self.edge_index = edge_index
            self.num_nodes = num_nodes
        else:
            raise NotImplementedError()


    @staticmethod
    @torch.no_grad()
    def get_adj(num_nodes:int, edge_index:torch.Tensor) -> torch.Tensor:
        adj = edge_index.new_zeros((num_nodes, num_nodes), dtype=torch.float32)
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        return adj
    
    @staticmethod
    @torch.no_grad()
    def get_edge_index(adj:torch.Tensor) -> torch.Tensor:
        src, dst = adj.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)

        new_edges = [
            (17, 16), (17, 18),
            (18, 13), (18, 17),
            (19, 3),  (19, 20),
            (20, 6),  (20, 19),
            (21, 14), (21, 15), (21, 22),
            (22, 11), (22, 12), (22, 21),
            (23, 2), (23, 5)
        ]

        bidirectional_edges = []
        for u, v in new_edges:
            bidirectional_edges.append((u, v))
            bidirectional_edges.append((v, u))

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
    print(L == L.t())
    
    eigval, eigvec = eigh(L) # eigen-decomposition
    sorted_index = torch.argsort(eigval)
    eigvec = eigvec[:, sorted_index[1:]] # [N, 16] positional encoding
    print(eigval)
    return eigvec


@torch.no_grad()
def ricci_rewire(
    edge_index: torch.LongTensor,          # [2, E]
    num_nodes: int,
    *,
    alpha: float = 0.6,                    # Wasserstein mixing (0.5~0.8 일반적)
    n_iter: int = 12,                      # Ricci-flow step
    step: float = 0.1,
    add_pct: float = 10.0,                 # 음의 곡률 하위 add_pct% 간선 추가
    drop_pct: float = 0.0,                 # 양의 곡률 상위 drop_pct% 간선 제거
    keep_self_loops: bool = True,
    device =  None,
):
    """
    Returns
    -------
    edge_index_new : [2, E']
    edge_weight_new: [E']   (1.0 for un-weighted edges)
    """
    # 1) edge_index → NetworkX 그래프 (무가중치 undirected)
    src, dst = edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    if keep_self_loops:
        G.add_edges_from([(i, i) for i in range(num_nodes)])

    # 2) Ollivier-Ricci curvature & flow
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO")
    orc.compute_ricci_curvature()                # 한번에 곡률만

    
    edge2curv = orc.G.edges   # dict: {(u,v):{'ricciCurvature':c}}
    curv_vals = np.array([d["ricciCurvature"] for d in edge2curv.values()])

    # 3) 임계값 계산
    add_thr  = np.percentile(curv_vals, add_pct)    # 낮을수록 더 negative
    drop_thr = np.percentile(curv_vals, 100 - drop_pct) if drop_pct > 0 else np.inf

    # 4) 간선 선택
    add_edges  = [(u, v) for (u, v), d in edge2curv.items() if d["ricciCurvature"] < add_thr]
    drop_edges = {(u, v) if u <= v else (v, u)          # 정렬하여 set 비교
                  for (u, v), d in edge2curv.items() if d["ricciCurvature"] > drop_thr}

    # 원본 간선에서 drop 대상 제거
    base_edges = {(u, v) if u <= v else (v, u) for u, v in G.edges()}
    base_edges -= drop_edges
    base_edges |= {(u, v) if u <= v else (v, u) for u, v in add_edges}  # 추가

    # 5) tensor로 변환
    src_new, dst_new = zip(*base_edges)
    edge_index_new = torch.tensor([src_new, dst_new], dtype=torch.long,
                                  device=device or edge_index.device)
    edge_weight_new = torch.ones(edge_index_new.shape[1],
                                 dtype=torch.float32, device=edge_index_new.device)
    return edge_index_new, edge_weight_new


# ──────────────────────────── usage ────────────────────────────
if __name__ == "__main__":

    x = CachedGraph.build_human_graph()

    ei, ew = ricci_rewire(x.edge_index, num_nodes=17+7,
                          alpha=0.7, add_pct=50.0, drop_pct=0.0, device='cuda')
    y = CachedGraph.get_adj(17+7, ei)
    print(f"rewired |E'| = {y[0:4]}")

