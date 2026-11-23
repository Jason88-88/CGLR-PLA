
import numpy as np
import pandas as pd
METRIC_NAMES = ["CI", "Pearson r", "MAE", "RMSE", "SD", "Spearman rho"]

def mse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mse_score(y_true, y_pred)))


def mae_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2 or np.std(y_true) < eps or np.std(y_pred) < eps:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sd_score(y_true, y_pred, eps=1e-12):
    """CASF-style residual SD of linear fit y_true = a * y_pred + b."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    if n < 2:
        return float("nan")

    pred_centered = y_pred - y_pred.mean()
    denom = np.sum(pred_centered ** 2)
    if denom < eps:
        return float("nan")

    slope = np.sum(pred_centered * (y_true - y_true.mean())) / denom
    intercept = y_true.mean() - slope * y_pred.mean()
    residual = y_true - (slope * y_pred + intercept)
    return float(np.sqrt(np.sum(residual ** 2) / (n - 1)))


def spearman_score(y_true, y_pred):
    y_true = pd.Series(np.asarray(y_true, dtype=np.float64).ravel()).rank(method="average").to_numpy()
    y_pred = pd.Series(np.asarray(y_pred, dtype=np.float64).ravel()).rank(method="average").to_numpy()
    return pearson_r_score(y_true, y_pred)


def concordance_index(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2:
        return float("nan")

    i, j = np.triu_indices(len(y_true), k=1)
    diff_true = y_true[i] - y_true[j]
    diff_pred = y_pred[i] - y_pred[j]
    valid = np.abs(diff_true) > eps
    if not np.any(valid):
        return float("nan")

    diff_true = diff_true[valid]
    diff_pred = diff_pred[valid]
    correct = np.sum(diff_true * diff_pred > eps)
    tied = np.sum(np.abs(diff_pred) <= eps)
    return float((correct + 0.5 * tied) / len(diff_true))


def compute_metrics(y_true, y_pred):
    return {
        "CI": concordance_index(y_true, y_pred),
        "Pearson r": pearson_r_score(y_true, y_pred),
        "MAE": mae_score(y_true, y_pred),
        "RMSE": rmse_score(y_true, y_pred),
        "SD": sd_score(y_true, y_pred),
        "Spearman rho": spearman_score(y_true, y_pred),
    }