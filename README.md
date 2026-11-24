# CGLR-PLA: Chemistry-Guided Local Rewiring and Geometry-Aware Interaction Representation for Protein–Ligand Binding Affinity Prediction
<img width="4323" height="4409" alt="image" src="https://github.com/user-attachments/assets/2755f9f4-862c-4725-a6f7-7c5d75b85c78" />

## Highlights
1. **Chemistry-Guided Local Rewiring (CGLR).** 
2. **Geometry-aware graph encoding.** 
3. **Decoupled molecular and interaction readout.** 

## Data
The PDBbind dataset can be downloaded [here](https://www.pdbbind-plus.org.cn/)

The CSAR-HIQ dataset can be downloaded [here](https://drive.google.com/file/d/1NGhylymFfNDLWiLuGBRFtO6U-JfhNmyG/view)

## Dependencies

- python >= 3.10
- pytorch >= 2.6.0 (CUDA 12.4)
- torch-geometric >= 2.7.0
- rdkit >= 2026.3.1
- numpy, pandas, scipy, pyarrow

## How to Run the Code

### 1. Build the graphs

```bash
python graph_construction.py --data-root /path/to/structures --output-dir /path/to/graphs \
    --split train /path/to/train_index.parquet \
    --split val /path/to/val_index.parquet \
    --split CASF_2016_core /path/to/casf2016_index.parquet
```

### 2. Train

```bash
python main.py --graph-dir /path/to/graphs --output-dir runs/seed42 --seed 42
```

### 3. Evaluate

```bash
python test.py --checkpoint runs/seed42/best_model.pt --graph-dir /path/to/graphs \
    --split CASF2016 --output-dir results/seed42
```

`--split` can be `CASF2016`, `CASF2013` or `CSAR_HiQ`.

## Citation

A citation will be added once the corresponding paper is published.
