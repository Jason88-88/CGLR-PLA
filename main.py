import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import gc
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import ContactOnlyPDBbindDataset, make_loader, move_batch_to_device
from metrics import METRIC_NAMES, compute_metrics, mse_score
from model import CGLRPLA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def config_to_dict(cfg):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}


def load_split(cfg, split):
    return ContactOnlyPDBbindDataset(
        split=split,
        graph_dir=cfg.graph_dir,
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        preload_graph_parts=cfg.preload_graph_parts,
        graph_part_cache_size=cfg.graph_part_cache_size,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch["graph"])
        loss = criterion(pred, batch["y"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = batch["y"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels, pdb_codes = [], [], []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = model(batch["graph"])
        loss = criterion(pred, batch["y"])

        bs = batch["y"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds.append(pred.cpu().view(-1))
        labels.append(batch["y"].cpu().view(-1))
        pdb_codes.extend(batch["pdb_codes"])

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()

    metrics = compute_metrics(y_true, y_pred)
    metrics["mse"] = mse_score(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples

    pred_df = pd.DataFrame({
        "PDB_code": pdb_codes,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred),
        "squared_error": (y_true - y_pred) ** 2,
    })
    return metrics, pred_df


def run(cfg):
    set_seed(cfg.seed)

    best_ckpt = cfg.output_dir / "best_model.pt"
    last_ckpt = cfg.output_dir / "last_model.pt"
    if not cfg.overwrite and (best_ckpt.exists() or last_ckpt.exists()):
        raise FileExistsError(f"Checkpoints already exist in {cfg.output_dir}. Pass --overwrite to replace.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CGLR-PLA | device={cfg.device} | seed={cfg.seed} | out={cfg.output_dir}")

    train_set = load_split(cfg, cfg.train_split)
    val_set = load_split(cfg, cfg.val_split)
    test_set = load_split(cfg, cfg.test_split) if cfg.test_split else None

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = make_loader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=gen)
    val_loader = make_loader(val_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = make_loader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) if test_set else None

    model = CGLRPLA(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        hidden_dim=cfg.hidden_dim,
        egnn_layers=cfg.egnn_layers,
        dropout_gnn=cfg.dropout_gnn,
        dropout_mlp=cfg.dropout_mlp,
    ).to(cfg.device)

    total_params, trainable_params = count_parameters(model)
    enc_params, _ = count_parameters(model.encoder)
    pred_params, _ = count_parameters(model.predictor)
    print(f"Parameters: total={total_params:,} (encoder={enc_params:,}, predictor={pred_params:,})")

    cfg_dict = config_to_dict(cfg)
    pd.DataFrame([{
        "model": "CGLRPLA",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "encoder_parameters": enc_params,
        "predictor_parameters": pred_params,
    }]).to_csv(cfg.output_dir / "model_parameter_summary.csv", index=False)
    pd.DataFrame([cfg_dict]).to_csv(cfg.output_dir / "run_config.csv", index=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr
    )

    best_val_mse = float("inf")
    best_epoch = -1
    patience_counter = 0
    log_rows = []
    log_path = cfg.output_dir / "training_log.csv"

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, cfg.grad_clip)
        val_metrics, val_pred_df = evaluate(model, val_loader, criterion, cfg.device)

        scheduler.step(val_metrics["mse"])
        lr = optimizer.param_groups[0]["lr"]

        improved = val_metrics["mse"] < best_val_mse - cfg.min_delta
        if improved:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mse": best_val_mse,
                "config": cfg_dict,
            }, best_ckpt)
            val_pred_df.to_csv(cfg.output_dir / "best_val_predictions.csv", index=False)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "config": cfg_dict,
        }, last_ckpt)

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["RMSE"],
            "val_mae": val_metrics["MAE"],
            "val_r": val_metrics["Pearson r"],
            "val_sd": val_metrics["SD"],
            "val_spearman": val_metrics["Spearman rho"],
            "val_ci": val_metrics["CI"],
            "lr": lr,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "is_best": int(improved),
        })
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        print(
            f"Ep {epoch:03d} | Train {train_loss:.5f} | Val MSE {val_metrics['mse']:.5f} | "
            f"RMSE {val_metrics['RMSE']:.4f} | MAE {val_metrics['MAE']:.4f} | R {val_metrics['Pearson r']:.4f} | "
            f"SD {val_metrics['SD']:.4f} | rho {val_metrics['Spearman rho']:.4f} | CI {val_metrics['CI']:.4f} | "
            f"LR {lr:.1e} | Best {best_val_mse:.5f}@{best_epoch} | Pat {patience_counter}/{cfg.early_stop_patience}"
        )

        if patience_counter >= cfg.early_stop_patience:
            print("Early stopping.")
            break

    if test_loader is not None:
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, test_pred_df = evaluate(model, test_loader, criterion, cfg.device)
        test_pred_df.to_csv(cfg.output_dir / "test_predictions.csv", index=False)
        pd.DataFrame([{k: test_metrics[k] for k in METRIC_NAMES}]).to_csv(
            cfg.output_dir / "final_test_metrics.csv", index=False
        )
        print(f"\n[{cfg.test_split}] Best epoch {ckpt['epoch']}: " + " | ".join(
            f"{k}={test_metrics[k]:.4f}" for k in METRIC_NAMES
        ))

    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CGLR-PLA.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="CASF_2016_core")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--graph-part-cache-size", type=int, default=128)
    parser.add_argument("--no-preload", dest="preload_graph_parts", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    cfg = parser.parse_args(argv)

    if str(cfg.test_split).lower() == "none":
        cfg.test_split = None

    cfg.node_dim = 108
    cfg.edge_dim = 24
    cfg.hidden_dim = 256
    cfg.egnn_layers = 3
    cfg.dropout_gnn = 0.1
    cfg.dropout_mlp = 0.2
    return cfg


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()