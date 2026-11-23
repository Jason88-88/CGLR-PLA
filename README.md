# CGLR-PLA

Protein-ligand binding affinity prediction on a single chemistry-rewired
complex graph. CGLR-PLA has three components:

1. **Chemistry-Guided Local Rewiring (CGLR).** For every ligand heavy atom,
   pocket heavy atoms within 5.0 A are collected, five chemical interaction
   flags (hydrogen bond, hydrophobic, salt bridge, aromatic, van der Waals)
   are computed for each candidate pair, and the number of interaction edges
   is set to `k_i = min(6, N_compatible)`, where `N_compatible` is the number
   of chemically compatible candidates. The `k_i` spatially nearest pocket
   atoms are then connected in both directions. The chemistry decides *how
   many* edges an atom gets; geometry decides *which* ones.
2. **Geometry-aware graph encoding.** Three invariant message-passing layers
   over fixed 3D coordinates. Messages are formed from the two node
   embeddings, the 24-D edge features and the interatomic distance, summed
   per node, then applied as a residual update with LayerNorm. Coordinates are
   never updated.
3. **Decoupled molecular and interaction readout.** Ligand and pocket atoms
   are pooled separately (mean + max), while the CGLR contact edges are pooled
   with an attention head normalised inside each complex. The five vectors are
   fused and passed to a regression head.

## Repository structure

```
CGLR-PLA/
├── README.md
├── environment.yaml        # conda environment (cglr-pla)
├── model.py                # CGLRPLA: encoder, contact attention, readout, head
├── dataset.py              # graph dataset, contact-aware batching, DataLoader
├── main.py                 # training / validation / early stopping
├── test.py                 # evaluation of a checkpoint on a benchmark split
├── metrics.py              # CI, Pearson r, MAE, RMSE, SD, Spearman rho
├── graph_construction.py   # CGLR graph builder (RDKit + multiprocessing)
└── tests.py                # consistency checks against the original scripts
```

## Requirements

Python 3.10, PyTorch 2.6.0 (CUDA 12.4), PyTorch Geometric 2.7.0, RDKit,
NumPy, pandas, SciPy, PyArrow.

```bash
conda env create -f environment.yaml
conda activate cglr-pla
```

A CUDA GPU is recommended for training; all scripts also run on CPU with
`--device cpu`.

## Dataset preparation

PDBbind v2020 is not redistributed here. Prepare, for every split, a
directory of structures and an index table:

```
<data-root>/<split>/<PDB>/<PDB>_pocket.pdb     # pocket, as released by PDBbind
<data-root>/<split>/<PDB>/<PDB>_ligand.sdf     # ligand
```

and one index file per split (parquet) with the columns `PDB_code` and
`log_Kd_Ki` (the binding affinity in log units). The splits used in the paper
are `train`, `val`, `CASF_2016_core`, `CASF_2013_core` and `CSAR_HiQ`.

## Graph construction

```bash
python graph_construction.py \
    --data-root /path/to/structures \
    --output-dir /path/to/graphs \
    --split train /path/to/train_index.parquet \
    --split val /path/to/val_index.parquet \
    --split CASF_2016_core /path/to/casf2016_index.parquet
```

Pocket preprocessing removes water and hydrogens and resolves altlocs
(`REMOVE_WATER`, `REMOVE_POCKET_H`, `RESOLVE_ALTLOC`); ligand hydrogens are
kept but excluded from interaction edges (`REMOVE_LIGAND_H = False`,
`EXCLUDE_H_IN_INTERACTION_EDGES = True`). Nodes carry 108-D atom features,
edges 24-D features (16-D Gaussian RBF over 0-5 A with sigma 0.5 A, 5 chemical
flags, 3-D edge type).

Each split produces

* `pd_graphs_<split>_part<N>.pt` — lists of PyTorch Geometric `Data` objects
  (100 graphs per file) with `x`, `pos`, `edge_index`, `edge_attr`, `y`,
  `pdb_code`, `n_pocket`, `n_drug`, `contact_pocket`, `contact_ligand`,
  `contact_edge_attr`;
* `pd_graphs_<split>_meta.csv` — one row per graph with `PDB_code`,
  `log_Kd_Ki`, `pt_file`, `part`, `index_in_part` and per-graph edge
  statistics;
* `pd_graphs_<split>_failed.csv` and `PDBbind_<split>_success_index.parquet`.

Samples are loaded in `(part, index_in_part)` order, i.e. the order in which
the graphs were written. Because the builder is multiprocessed, rerunning it
can produce a different order; keep the generated files to keep an order.

## Training

```bash
python main.py \
    --graph-dir /path/to/graphs \
    --output-dir runs/seed42 \
    --seed 42 \
    --max-epochs 500
```

AdamW with learning rate 7e-5 and weight decay 1e-5, MSE loss, batch size 32,
gradient clipping 5.0, `ReduceLROnPlateau` (factor 0.5, patience 10,
min_lr 1e-6) on the validation MSE, early stopping after 20 epochs without a
1e-6 improvement. The checkpoint with the lowest validation MSE is written to
`best_model.pt` (fields `epoch`, `model_state_dict`, `optimizer_state_dict`,
`best_val_mse`, `config`), alongside `last_model.pt`, `training_log.csv`,
`best_val_predictions.csv`, `run_config.csv` and `model_parameter_summary.csv`.
The seeds reported in the paper are 42, 3047 and 114514.

The default maximum is 500 epochs, as reported in the paper. The original
research script used `max_epochs = 200`; pass `--max-epochs 200` to reproduce
that setting. Every published run stopped early (between epoch 25 and 62),
well before either limit.

Run `python main.py --help` for the remaining options (`--device`,
`--batch-size`, `--train-split`, `--val-split`, `--test-split`, the optimizer
and scheduler settings, `--no-preload`).

## Evaluation

```bash
python test.py \
    --checkpoint runs/seed42/best_model.pt \
    --graph-dir /path/to/graphs \
    --split CASF2016 \
    --output-dir results/seed42
```

`--split` accepts `CASF2016`, `CASF2013`, `CSAR_HiQ` or a raw split name. The
checkpoint is loaded with `strict=True`, the model is run under `eval()` and
`torch.no_grad()`, and the sample order of the metadata is preserved. Two
files are written: `<name>_predictions.csv` (`PDB_code`, `y_true`, `y_pred`,
`abs_error`, `squared_error`) and `<name>_metrics.csv` (CI, Pearson r, MAE,
RMSE, SD, Spearman rho). Missing graph files or metadata are reported by name
instead of being regenerated.

To evaluate a subset of a split — for example the 88 CSAR-HiQ complexes whose
PDB IDs do not occur in the training or validation sets — pass a metadata CSV
with the same columns:

```bash
python test.py --checkpoint runs/seed42/best_model.pt \
    --graph-dir /path/to/csar_hiq_graphs --split CSAR_HiQ \
    --meta-csv /path/to/CSAR_HiQ_exact_clean.csv \
    --name CSAR_HiQ_88 --output-dir results/seed42
```

## Pretrained model

Checkpoints are not included in this repository. `test.py` works with any
`best_model.pt` written by `main.py`; `CGLRPLA()` uses the published
architecture (108-D nodes, 24-D edges, hidden size 256, 3 EGNN layers,
2,650,114 parameters), so a checkpoint trained with it loads without
conversion.

## Consistency checks

`tests.py` compares this code against the original research scripts
(`CGLR_EGNN.py`, `CGLR_graph_builder.py`) on existing graph files and
checkpoints: module structure and `state_dict` keys, `strict=True` checkpoint
loading, dataset order and contact batching offsets, per-sample prediction
equality, the six metric formulas, and a static comparison of the graph
construction rules. Nothing is rebuilt or retrained.

```bash
python tests.py --original-dir /path/to/original/scripts \
    --graph-dir /path/to/graphs \
    --checkpoint runs/seed42/best_model.pt
```

Checks whose inputs are absent are reported as `SKIP`.

## Citation

A citation will be added once the corresponding paper is published.
