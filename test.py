import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from dataset import ContactOnlyPDBbindDataset, make_loader
from main import evaluate, set_seed
from metrics import METRIC_NAMES
from model import CGLRPLA

SPLIT_ALIASES = {
    "CASF2016": "CASF_2016_core",
    "CASF2013": "CASF_2013_core",
    "CSAR_HiQ": "CSAR_HiQ",
}


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CGLRPLA().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def predict_split(checkpoint_path, graph_dir, split, device, batch_size=32, num_workers=0, meta_path=None):
    dataset = ContactOnlyPDBbindDataset(split=split, graph_dir=graph_dir, meta_path=meta_path)
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model, ckpt = load_model(checkpoint_path, device)
    metrics, pred_df = evaluate(model, loader, nn.MSELoss(), device)
    return metrics, pred_df, ckpt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a CGLR-PLA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--meta-csv", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    split = SPLIT_ALIASES.get(args.split, args.split)
    name = args.name or split

    pred_path = args.output_dir / f"{name}_predictions.csv"
    metrics_path = args.output_dir / f"{name}_metrics.csv"
    if not args.overwrite and (pred_path.exists() or metrics_path.exists()):
        sys.exit(f"Output files already exist in {args.output_dir}. Pass --overwrite to replace.")

    set_seed(42)
    metrics, pred_df, ckpt = predict_split(
        args.checkpoint,
        args.graph_dir,
        split,
        args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        meta_path=args.meta_csv,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    pd.DataFrame([{
        "split": split,
        "n_samples": len(pred_df),
        "checkpoint": str(args.checkpoint),
        **{k: metrics[k] for k in METRIC_NAMES},
    }]).to_csv(metrics_path, index=False)

    print(f"\n[{name}] n={len(pred_df)} (epoch {ckpt.get('epoch')}): " + " | ".join(
        f"{k}={metrics[k]:.4f}" for k in METRIC_NAMES
    ))
    print(f"Saved: {pred_path}, {metrics_path}")


if __name__ == "__main__":
    main()