import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

class InvariantEGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pos, edge_index, edge_attr):
        src, dst = edge_index
        dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True).to(h.dtype)

        msg = self.message_mlp(torch.cat([h[src], h[dst], edge_attr.to(h.dtype), dist], dim=-1))
        agg = h.new_zeros(h.size(0), h.size(-1)).index_add_(0, dst, msg)

        update = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + self.dropout(update))


class EGNNEncoder(nn.Module):
    def __init__(self, in_dim=108, edge_dim=24, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([
            InvariantEGNNLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # [h_lig, h_poc, |h_lig - h_poc|, h_lig * h_poc, e_contact]
        self.contact_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.contact_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Fusion of [lig_mean, lig_max, poc_mean, poc_max, contact]
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _masked_mean_max_pool(h, batch, mask, num_graphs):
        if not mask.any():
            zeros = h.new_zeros(num_graphs, h.size(-1))
            return zeros, zeros

        sel_h = h[mask]
        sel_batch = batch[mask]
        mean_vec = global_mean_pool(sel_h, sel_batch, size=num_graphs)
        max_vec = global_max_pool(sel_h, sel_batch, size=num_graphs)

        present = torch.zeros(num_graphs, dtype=torch.bool, device=h.device)
        present[sel_batch] = True
        max_vec = torch.where(present.unsqueeze(-1), max_vec, torch.zeros_like(max_vec))
        return mean_vec, max_vec

    def _contact_endpoint_pool(self, h, graph):
        num_graphs = graph.num_graphs
        if getattr(graph, "contact_pocket", None) is None or graph.contact_pocket.numel() == 0:
            return h.new_zeros(num_graphs, self.hidden_dim)

        pocket_h = h[graph.contact_pocket]
        ligand_h = h[graph.contact_ligand]
        contact_input = torch.cat(
            [ligand_h, pocket_h, torch.abs(ligand_h - pocket_h), ligand_h * pocket_h, graph.contact_edge_attr.to(h.dtype)],
            dim=-1,
        )
        contact_h = self.contact_mlp(contact_input)
        attn = softmax(self.contact_attn(contact_h), graph.contact_batch)
        return global_add_pool(contact_h * attn, graph.contact_batch, size=num_graphs)

    def forward(self, graph):
        h = self.input_proj(graph.x.float())
        pos = graph.pos.float()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr.float()
        batch = graph.batch

        for layer in self.layers:
            h = layer(h, pos, edge_index, edge_attr)

        num_graphs = graph.num_graphs
        n_pocket = torch.as_tensor(graph.n_pocket, device=h.device, dtype=torch.long).view(-1)

        # Pocket atoms occupy the first n_pocket nodes of each graph
        local_idx = torch.arange(h.size(0), device=h.device) - graph.ptr[batch]
        pocket_mask = local_idx < n_pocket[batch]

        lig_mean, lig_max = self._masked_mean_max_pool(h, batch, ~pocket_mask, num_graphs)
        poc_mean, poc_max = self._masked_mean_max_pool(h, batch, pocket_mask, num_graphs)
        contact_vec = self._contact_endpoint_pool(h, graph)

        pooled = torch.cat([lig_mean, lig_max, poc_mean, poc_max, contact_vec], dim=-1)
        return self.pool_norm(self.pool_mlp(pooled))


class CGLRPLA(nn.Module):
    def __init__(
        self,
        node_dim=108,
        edge_dim=24,
        hidden_dim=256,
        egnn_layers=3,
        dropout_gnn=0.1,
        dropout_mlp=0.2,
    ):
        super().__init__()
        self.encoder = EGNNEncoder(
            in_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=egnn_layers,
            dropout=dropout_gnn,
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(256, 1),
        )

    def forward(self, graph):
        return self.predictor(self.encoder(graph))