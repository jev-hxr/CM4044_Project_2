"""Model architectures: multi-task MLP on ECFP4 and a GINE graph network."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(1024, 256), dropout=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GIN(nn.Module):
    """GINE (Hu et al. 2020) with summed categorical atom/bond embeddings,
    mean+max pooling and a 2-layer prediction head."""

    def __init__(self, n_out, hidden=128, layers=4, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import GINEConv
        from torch_geometric.utils.smiles import e_map, x_map

        self.atom_emb = nn.ModuleList(nn.Embedding(len(v), hidden) for v in x_map.values())
        self.bond_emb = nn.ModuleList(
            nn.ModuleList(nn.Embedding(len(v), hidden) for v in e_map.values()) for _ in range(layers))
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.BatchNorm1d(2 * hidden),
                                   nn.ReLU(), nn.Linear(2 * hidden, hidden)), train_eps=True)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))

    def forward(self, data):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = sum(emb(data.x[:, i]) for i, emb in enumerate(self.atom_emb))
        for conv, norm, bond_emb in zip(self.convs, self.norms, self.bond_emb):
            e = sum(emb(data.edge_attr[:, i]) for i, emb in enumerate(bond_emb))
            h = h + self.dropout(torch.relu(norm(conv(h, data.edge_index, e))))  # residual
        g = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=1)
        return self.head(g)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
