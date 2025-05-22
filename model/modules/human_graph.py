import torch

class CachedGraph():
    @torch.no_grad()
    def __init__(self, edge_index:torch.Tensor, num_nodes:int, mode:str):

        if mode == 'skeleton':
            one_hop_adj = CachedGraph.get_adj(num_nodes, edge_index)
            two_hop_adj = one_hop_adj @ one_hop_adj
            adj = one_hop_adj + two_hop_adj
            adj.fill_diagonal_(0)
            self.edge_index = self.get_edge_index(adj)

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
        return edge_index


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
            return CachedGraph(Cayley(17, device), 17, mode)

@torch.no_grad()
def Cayley(n:int, device):
    assert n >= 3
    gens = [2, 3]
    edges = []
    for v in range(n):
        for g in gens:
            edges.append([v, (v + g) % n])
            edges.append([v, (v - g) % n])

    edge_index_cayley = torch.tensor(edges, dtype=torch.long, device=device).t()
    return edge_index_cayley
