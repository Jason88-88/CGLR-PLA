import random
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

NODE_DIM = 108
EDGE_DIM = 24
META_COLUMNS = {"PDB_code", "pt_file", "part", "index_in_part"}
EXCLUDE_KEYS = [
    "contact_pocket", "contact_ligand", "contact_edge_attr", "contact_flags",
    "contact_batch", "triplet_edge_id", "triplet_outer", "triplet_center",
    "triplet_other", "triplet_side", "triplet_attr",
]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch_to_device(batch, device):
    batch["graph"] = batch["graph"].to(device, non_blocking=True)
    batch["y"] = batch["y"].to(device, non_blocking=True).float()
    return batch


class ContactOnlyPDBbindDataset(Dataset):
    def __init__(
        self,
        split,
        graph_dir,
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        preload_graph_parts=True,
        graph_part_cache_size=128,
        validate_graph_pairing=False,
        meta_path=None,
    ):
        self.split = split
        self.graph_dir = Path(graph_dir)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_part_cache_size = graph_part_cache_size
        self.validate_graph_pairing = validate_graph_pairing
        self.meta_path = Path(meta_path) if meta_path else self.graph_dir / f"pd_graphs_{split}_meta.csv"

        meta = pd.read_csv(self.meta_path)
        missing = META_COLUMNS - set(meta.columns)
        if missing:
            raise ValueError(f"{self.meta_path} missing columns: {sorted(missing)}")

        meta["PDB_code"] = meta["PDB_code"].astype(str).str.strip()
        meta["PDB_code_norm"] = meta["PDB_code"].str.lower()
        assert not meta["PDB_code_norm"].duplicated().any(), f"Duplicate PDB codes in {self.meta_path}"

        self.meta = meta.sort_values(["part", "index_in_part"]).reset_index(drop=True)
        self.part_cache = OrderedDict()

        if preload_graph_parts:
            for pt_file in sorted(self.meta["pt_file"].astype(str).unique()):
                self.part_cache[pt_file] = torch.load(
                    self.graph_dir / pt_file, map_location="cpu", weights_only=False
                )

    def __len__(self):
        return len(self.meta)

    def _load_part(self, pt_file):
        if pt_file in self.part_cache:
            self.part_cache.move_to_end(pt_file)
            return self.part_cache[pt_file]

        graphs = torch.load(self.graph_dir / pt_file, map_location="cpu", weights_only=False)
        self.part_cache[pt_file] = graphs
        if len(self.part_cache) > self.graph_part_cache_size:
            self.part_cache.popitem(last=False)
        return graphs

    def _validate_graph(self, graph, expected_code):
        graph_code = str(getattr(graph, "pdb_code", "")).strip().lower()
        if graph_code and graph_code != expected_code:
            raise ValueError(f"PDB mismatch: meta={expected_code}, graph={graph_code}")
        if graph.x.size(-1) != self.node_dim or graph.edge_attr.size(-1) != self.edge_dim:
            raise ValueError(f"Feature dim mismatch for {expected_code}")

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        graphs = self._load_part(str(row["pt_file"]))
        graph = graphs[int(row["index_in_part"])]

        if self.validate_graph_pairing:
            self._validate_graph(graph, row["PDB_code_norm"])

        return {
            "pdb_code": row["PDB_code"],
            "graph": graph,
            "y": graph.y.view(-1).float(),
        }


def batch_graph_list_for_egnn(graphs):
    batch_graph = Batch.from_data_list(graphs, exclude_keys=EXCLUDE_KEYS)

    pocket, ligand, attributes, contact_batch = [], [], [], []
    node_offset = 0

    for graph_id, graph in enumerate(graphs):
        contacts = getattr(graph, "contact_pocket", None)
        num_contacts = 0 if contacts is None else int(contacts.numel())
        if num_contacts > 0:
            pocket.append(contacts.long() + node_offset)
            ligand.append(graph.contact_ligand.long() + node_offset)
            attributes.append(graph.contact_edge_attr.float())
            contact_batch.append(torch.full((num_contacts,), graph_id, dtype=torch.long))
        node_offset += int(graph.x.size(0))

    if pocket:
        batch_graph.contact_pocket = torch.cat(pocket)
        batch_graph.contact_ligand = torch.cat(ligand)
        batch_graph.contact_edge_attr = torch.cat(attributes)
        batch_graph.contact_batch = torch.cat(contact_batch)
    else:
        edge_dim = int(graphs[0].edge_attr.size(1))
        batch_graph.contact_pocket = torch.empty(0, dtype=torch.long)
        batch_graph.contact_ligand = torch.empty(0, dtype=torch.long)
        batch_graph.contact_edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
        batch_graph.contact_batch = torch.empty(0, dtype=torch.long)

    return batch_graph


def collate_fn(samples):
    return {
        "pdb_codes": [s["pdb_code"] for s in samples],
        "graph": batch_graph_list_for_egnn([s["graph"] for s in samples]),
        "y": torch.stack([s["y"] for s in samples]).view(-1, 1),
    }


def make_loader(dataset, batch_size=32, shuffle=False, num_workers=0, generator=None):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )