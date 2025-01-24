from torch import nn
import torch
import torch.nn.functional as F
from aviary.roost.model import DescriptorNetwork


class CompositionEmbedding(torch.nn.Module):
    def __init__(
        self,
        elem_input_dim: int = 16,
        elem_hidden_dim: int = 64,
        comp_embed_dim: int = 64,
    ):
        super().__init__()
        self.gnn = DescriptorNetwork(
            elem_emb_len=elem_input_dim, elem_fea_len=elem_hidden_dim, n_graph=1
        )
        self.head = nn.Linear(elem_hidden_dim, comp_embed_dim)
        self.rescale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def embed(self, X):
        return self.head(self.gnn(*X))

    def forward(self, X1, X2):
        z1 = self.embed(X1)
        z2 = self.embed(X2)

        # dists = torch.sqrt(torch.sum(torch.square(z1 - z2), axis=1))
        dists = torch.linalg.vector_norm(z1 - z2, dim=1)
        return self.to_probability(dists)

    def to_probability(self, dists):
        return torch.tanh(torch.clamp(dists * torch.exp(self.rescale), 0.0, 5.0))
