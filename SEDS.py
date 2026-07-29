from __future__ import annotations

"""
Sensitivity-Engine Derivative Surrogates
Greek- and Structure-Consistent Supervision

The implementation extends the convergence horizon and adds explicit end-to-end progress, ETA, and stopping diagnostics.

This revision removes all local-region training emphasis. The short-maturity,
near-the-money slice is retained only as an ex-post high-curvature diagnostic.

Main matched models
-------------------
Model A: scalar normalized-price potential c=C/S; all reported sensitivities are
         recovered from automatic derivatives of c.
Model B: direct heads for normalized price and selected first-/second-order
         sensitivities.
Model C: Model B plus price-potential consistency between direct heads and
         automatic derivatives of the normalized-price head.
Model D: Model C plus a two-leg BSM/DSE decomposition with supervised digital
         weights and price reconstruction.

Scientific-computing controls
-----------------------------
* point-in-time backward as-of alignment with source-age checks;
* float64 teacher arithmetic by default;
* stable normal CDF/PDF and exponential evaluation;
* standardized state and target coordinates;
* smooth SiLU networks so second automatic derivatives are meaningful;
* fourth-order finite-difference audits with machine-epsilon step scales;
* analytic decomposition and Greek identity checks;
* common architecture, optimizer, stopping rule, validation score, and seeds;
* deterministic seed handling, gradient clipping, and learning-rate reduction;
* paper-facing CSV and LaTeX outputs generated from the completed run.

Required local files
--------------------
spx_stooq.csv, DGS3MO.csv, DGS2.csv, DGS10.csv, VIXCLS.csv, VXVCLS.csv.
They are read from the script directory unless --data-dir is supplied.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import t as student_t_dist
from scipy.stats import ttest_1samp
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import TensorDataset

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
TAU_GRID_DAYS = np.array([3, 5, 7, 14, 21, 30, 45, 60, 90], dtype=float)
TAU_GRID_YEARS = TAU_GRID_DAYS / CALENDAR_DAYS_PER_YEAR
K_GRID = np.array(
    [-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.0,
     0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15],
    dtype=float,
)
HIGH_CURVATURE_TAU_MAX = 30.0 / CALENDAR_DAYS_PER_YEAR
HIGH_CURVATURE_K_MAX = 0.05

LOCAL_SOURCE_FILENAMES = {
    "spx": "spx_stooq.csv",
    "DGS3MO": "DGS3MO.csv",
    "DGS2": "DGS2.csv",
    "DGS10": "DGS10.csv",
    "VIXCLS": "VIXCLS.csv",
    "VXVCLS": "VXVCLS.csv",
}

FEATURE_COLS = ["k", "tau", "r_tau", "sigma_teacher"]
TARGET_COLS = [
    "price_teacher_normalized",
    "delta_teacher",
    "spot_gamma_teacher_normalized",
    "vega_teacher_normalized",
    "vanna_teacher",
    "volga_teacher_normalized",
]
TARGET_LABELS = ["price_norm", "delta", "spot_gamma", "vega_norm", "vanna", "volga_norm"]
MODEL_NAMES = ["Model_A", "Model_B", "Model_C", "Model_D"]
MODEL_PROGRESS_WEIGHTS = {"A": 1.0, "B": 0.65, "C": 2.55, "D": 3.25}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def default_data_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def human_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


@dataclass
class RunProgress:
    """Weighted run-level progress and approximate ETA reporting."""

    n_seeds: int
    start_time: float
    completed_work: float = 0.0
    completed_models: int = 0

    @property
    def total_models(self) -> int:
        return self.n_seeds * len(MODEL_NAMES)

    @property
    def total_work(self) -> float:
        return self.n_seeds * sum(MODEL_PROGRESS_WEIGHTS.values())

    def stage(self, label: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(f"[progress] stage={label} elapsed={human_duration(elapsed)}", flush=True)

    def epoch_suffix(self, kind: str, epoch: int, max_epochs: int, model_elapsed: float) -> str:
        model_fraction = min(max(epoch / max(max_epochs, 1), 0.0), 1.0)
        active_work = MODEL_PROGRESS_WEIGHTS[kind] * model_fraction
        overall_fraction = min((self.completed_work + active_work) / max(self.total_work, 1e-12), 1.0)
        elapsed = time.perf_counter() - self.start_time
        run_eta = elapsed * (1.0 - overall_fraction) / overall_fraction if overall_fraction > 0.0 else float("nan")
        epoch_rate = model_elapsed / max(epoch, 1)
        model_eta = epoch_rate * max(max_epochs - epoch, 0)
        return (
            f"model={100.0 * model_fraction:5.1f}% "
            f"overall~={100.0 * overall_fraction:5.1f}% "
            f"elapsed={human_duration(elapsed)} model_eta<={human_duration(model_eta)} "
            f"run_eta~={human_duration(run_eta)}"
        )

    def mark_model_complete(self, kind: str, label: str, *, cached: bool = False) -> None:
        self.completed_work = min(
            self.completed_work + MODEL_PROGRESS_WEIGHTS[kind], self.total_work
        )
        self.completed_models = min(self.completed_models + 1, self.total_models)
        fraction = self.completed_work / max(self.total_work, 1e-12)
        elapsed = time.perf_counter() - self.start_time
        eta = elapsed * (1.0 - fraction) / fraction if fraction > 0.0 else float("nan")
        source = "cache" if cached else "computed"
        print(
            f"[progress] models={self.completed_models}/{self.total_models} "
            f"overall~={100.0 * fraction:5.1f}% source={source} "
            f"last={label} elapsed={human_duration(elapsed)} run_eta~={human_duration(eta)}",
            flush=True,
        )

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(
            f"[progress] complete models={self.completed_models}/{self.total_models} "
            f"elapsed={human_duration(elapsed)}",
            flush=True,
        )


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_numpy_dtype(name: str):
    mapping = {"float64": np.float64, "float32": np.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return mapping[name]


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {"float64": torch.float64, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return mapping[name]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    if len(set(values)) != len(values):
        raise ValueError("Seeds must be unique.")
    return values


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def expected_shortfall_loss(pnl: np.ndarray, q: float = 0.95) -> float:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return float("nan")
    loss = -pnl
    threshold = float(np.quantile(loss, q))
    tail = loss[loss >= threshold]
    return float(np.mean(tail)) if tail.size else float("nan")


def safe_exp(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    finfo = np.finfo(dtype)
    return np.exp(np.clip(x, np.log(finfo.tiny), np.log(finfo.max)))


def norm_pdf(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    return np.exp(-0.5 * x * x) / np.sqrt(np.asarray(2.0 * np.pi, dtype=dtype))


def finite_difference_first_step(value: float, dtype=np.float64, requested: float = 1e-4) -> float:
    eps = np.finfo(dtype).eps
    return float(max(requested, eps ** (1.0 / 5.0) * max(1.0, abs(value))))


def finite_difference_second_step(value: float, dtype=np.float64, requested: float = 1e-3) -> float:
    eps = np.finfo(dtype).eps
    return float(max(requested, eps ** (1.0 / 6.0) * max(1.0, abs(value))))


def finite_difference_mixed_step(value: float, dtype=np.float64, requested: float = 1e-4) -> float:
    eps = np.finfo(dtype).eps
    return float(max(requested, eps ** 0.25 * max(1.0, abs(value))))


def stable_json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_file_fingerprint(data_dir: Path, use_mock: bool, n_mock_days: int, seed: int) -> Dict[str, Any]:
    if use_mock:
        return {"mode": "mock", "n_mock_days": int(n_mock_days), "seed": int(seed)}
    files: Dict[str, Any] = {}
    for filename in LOCAL_SOURCE_FILENAMES.values():
        path = data_dir / filename
        if not path.exists():
            files[filename] = {"missing": True}
        else:
            stat = path.stat()
            files[filename] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    return {"mode": "local_csv", "files": files}


def processed_cache_signature(args: argparse.Namespace, data_dir: Path, seed: int) -> Dict[str, Any]:
    return {
        "cache_version": 2,
        "inputs": input_file_fingerprint(data_dir, args.mock_data, args.n_mock_days, seed),
        "analysis_start": args.analysis_start,
        "analysis_end": args.analysis_end,
        "max_source_age_days": args.max_source_age_days,
        "teacher_dtype": args.dtype,
        "teacher_audit_samples": args.teacher_audit_samples,
        "tau_grid_days": TAU_GRID_DAYS.tolist(),
        "k_grid": K_GRID.tolist(),
        "high_curvature_tau_max": HIGH_CURVATURE_TAU_MAX,
        "high_curvature_k_max": HIGH_CURVATURE_K_MAX,
    }


def load_processed_cache(cache_dir: Path, signature: Dict[str, Any]) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    meta_path = cache_dir / "processed_cache_meta.json"
    paths = {
        "daily": cache_dir / "daily.pkl",
        "panel": cache_dir / "panel.pkl",
        "source_audit": cache_dir / "source_audit.pkl",
        "validation": cache_dir / "validation.pkl",
        "audit": cache_dir / "audit.pkl",
    }
    if not meta_path.exists() or any(not p.exists() for p in paths.values()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature_hash") != stable_json_hash(signature):
            return None
        return (
            pd.read_pickle(paths["daily"]),
            pd.read_pickle(paths["panel"]),
            pd.read_pickle(paths["source_audit"]),
            pd.read_pickle(paths["validation"]),
            pd.read_pickle(paths["audit"]),
        )
    except Exception as exc:
        warnings.warn(f"Ignoring unusable processed cache: {exc}", RuntimeWarning)
        return None


def save_processed_cache(
    cache_dir: Path,
    signature: Dict[str, Any],
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    source_audit: pd.DataFrame,
    validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    ensure_dir(cache_dir)
    daily.to_pickle(cache_dir / "daily.pkl")
    panel.to_pickle(cache_dir / "panel.pkl")
    source_audit.to_pickle(cache_dir / "source_audit.pkl")
    validation.to_pickle(cache_dir / "validation.pkl")
    audit.to_pickle(cache_dir / "audit.pkl")
    meta = {"signature_hash": stable_json_hash(signature), "signature": signature}
    (cache_dir / "processed_cache_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Point-in-time data construction
# ---------------------------------------------------------------------------

def read_local_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Place the six CSV inputs beside the script "
            "or pass --data-dir."
        )
    return pd.read_csv(path)


def fetch_stooq_spx(data_dir: Path) -> pd.DataFrame:
    path = data_dir / LOCAL_SOURCE_FILENAMES["spx"]
    df = read_local_csv(path)
    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError(f"{path.name} must contain Date and Close columns.")
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["SPX_Close"] = numeric_series(out["Close"])
    out = out.drop(columns=["Close"]).dropna()
    out = out[out["SPX_Close"] > 0.0]
    out = out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable observations in {path.name}.")
    return out


def fetch_fred_series(series_id: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / LOCAL_SOURCE_FILENAMES[series_id]
    df = read_local_csv(path)
    if len(df.columns) < 2:
        raise ValueError(f"{path.name} must have at least two columns.")
    out = df.iloc[:, :2].copy()
    out.columns = ["Date", series_id]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[series_id] = numeric_series(out[series_id])
    out = out.dropna().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable observations in {path.name}.")
    if series_id in {"VIXCLS", "VXVCLS"} and (out[series_id] <= 0.0).any():
        raise ValueError(f"{path.name} contains nonpositive volatility-index values.")
    return out


def build_mock_daily_data(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    ret = 0.00025 + 0.0105 * rng.standard_normal(n_days)
    spx = 1000.0 * np.exp(np.cumsum(ret))
    t = np.arange(n_days)
    dgs3mo = np.clip(1.0 + 0.7 * np.sin(t / 120.0) + 0.05 * rng.standard_normal(n_days), 0.05, None)
    dgs2 = np.clip(1.8 + 0.8 * np.sin(t / 180.0 + 0.4) + 0.05 * rng.standard_normal(n_days), 0.10, None)
    dgs10 = np.clip(2.7 + 0.7 * np.sin(t / 220.0 + 0.9) + 0.05 * rng.standard_normal(n_days), 0.20, None)
    vix = np.clip(18.0 + 4.0 * np.sin(t / 75.0) + 2.0 * np.maximum(0.0, rng.standard_normal(n_days)), 8.0, None)
    vxv = np.clip(20.0 + 3.0 * np.sin(t / 100.0 + 0.2) + 1.6 * np.maximum(0.0, rng.standard_normal(n_days)), 9.0, None)
    return pd.DataFrame({
        "Date": dates,
        "SPX_Close": spx,
        "DGS3MO": dgs3mo,
        "DGS2": dgs2,
        "DGS10": dgs10,
        "VIXCLS": vix,
        "VXVCLS": vxv,
    })


def load_daily_market_data(
    *,
    use_mock: bool,
    n_mock_days: int,
    seed: int,
    data_dir: Path,
    analysis_start: Optional[str],
    analysis_end: Optional[str],
    max_source_age_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if max_source_age_days < 0:
        raise ValueError("max_source_age_days must be nonnegative.")

    if use_mock:
        df = build_mock_daily_data(n_mock_days, seed).sort_values("Date").reset_index(drop=True)
        df["log_return"] = np.log(df["SPX_Close"]).diff()
        df["RV20"] = df["log_return"].rolling(20, min_periods=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        df["RV60"] = df["log_return"].rolling(60, min_periods=60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        audit = pd.DataFrame([{
            "source": "mock_joint_series",
            "raw_start": df["Date"].min(),
            "raw_end": df["Date"].max(),
            "raw_usable_rows": len(df),
            "common_start": df["Date"].min(),
            "common_end": df["Date"].max(),
            "max_observation_age_days": 0,
            "rows_over_age_limit": 0,
        }])
        return df, audit

    spx = fetch_stooq_spx(data_dir)
    spx["log_return"] = np.log(spx["SPX_Close"]).diff()
    spx["RV20"] = spx["log_return"].rolling(20, min_periods=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    spx["RV60"] = spx["log_return"].rolling(60, min_periods=60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    fred = {sid: fetch_fred_series(sid, data_dir) for sid in ["DGS3MO", "DGS2", "DGS10", "VIXCLS", "VXVCLS"]}
    common_start = max([spx["Date"].min()] + [frag["Date"].min() for frag in fred.values()])
    common_end = min([spx["Date"].max()] + [frag["Date"].max() for frag in fred.values()])

    if analysis_start is not None:
        requested = pd.Timestamp(analysis_start)
        if requested < common_start:
            raise ValueError(f"Requested start precedes common source start {common_start.date()}.")
        common_start = requested
    if analysis_end is not None:
        requested = pd.Timestamp(analysis_end)
        if requested > common_end:
            raise ValueError(f"Requested end exceeds common source end {common_end.date()}.")
        common_end = requested
    if common_start > common_end:
        raise ValueError("The requested/common date window is empty.")

    master = spx[(spx["Date"] >= common_start) & (spx["Date"] <= common_end)].copy().sort_values("Date")
    audit_rows: List[Dict[str, Any]] = [{
        "source": "spx",
        "raw_start": spx["Date"].min(),
        "raw_end": spx["Date"].max(),
        "raw_usable_rows": len(spx),
        "common_start": common_start,
        "common_end": common_end,
        "max_observation_age_days": 0,
        "rows_over_age_limit": 0,
    }]

    for sid, frag in fred.items():
        source_date_col = f"{sid}_source_date"
        right = frag.rename(columns={"Date": source_date_col}).sort_values(source_date_col)
        master = pd.merge_asof(
            master.sort_values("Date"),
            right,
            left_on="Date",
            right_on=source_date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        age = (master["Date"] - master[source_date_col]).dt.days
        invalid = age.isna() | (age > max_source_age_days)
        audit_rows.append({
            "source": sid,
            "raw_start": frag["Date"].min(),
            "raw_end": frag["Date"].max(),
            "raw_usable_rows": len(frag),
            "common_start": common_start,
            "common_end": common_end,
            "max_observation_age_days": int(age.dropna().max()) if age.notna().any() else -1,
            "rows_over_age_limit": int(invalid.sum()),
        })
        if invalid.any():
            examples = master.loc[invalid, "Date"].dt.strftime("%Y-%m-%d").head(5).tolist()
            raise ValueError(f"{sid} has stale/missing aligned observations. Examples: {examples}")

    return master.sort_values("Date").reset_index(drop=True), pd.DataFrame(audit_rows)


def prepare_daily_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    out["IV30"] = out["VIXCLS"] / 100.0
    out["IV90"] = out["VXVCLS"] / 100.0
    required = ["Date", "SPX_Close", "RV20", "RV60", "IV30", "IV90", "DGS3MO", "DGS2", "DGS10"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing daily-state columns: {missing}")
    out = out.dropna(subset=required).reset_index(drop=True)
    if out.empty:
        raise ValueError("No complete daily states remain.")
    if (out["SPX_Close"] <= 0.0).any():
        raise ValueError("SPX_Close must be positive.")
    if (out[["RV20", "RV60", "IV30", "IV90"]] <= 0.0).any().any():
        raise ValueError("Volatility inputs must be positive.")
    return out


# ---------------------------------------------------------------------------
# Teacher labels and structural identities
# ---------------------------------------------------------------------------

def piecewise_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, dtype=np.float64) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    xp = np.asarray(xp, dtype=dtype)
    fp = np.asarray(fp, dtype=dtype)
    y = np.interp(x, xp, fp)
    left = x < xp[0]
    right = x > xp[-1]
    if np.any(left):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left] = fp[0] + slope * (x[left] - xp[0])
    if np.any(right):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right] = fp[-1] + slope * (x[right] - xp[-1])
    return y


def piecewise_linear_weight_matrix(x: np.ndarray, xp: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Return linear interpolation/extrapolation weights for fixed knots.

    The matrix depends only on the requested maturities and knot locations, so
    it can be reused for every date.  This removes the date-by-date Python loop
    from teacher-panel construction while preserving the same piecewise-linear
    rule as :func:`piecewise_linear_extrap`.
    """
    x = np.asarray(x, dtype=dtype)
    xp = np.asarray(xp, dtype=dtype)
    if xp.ndim != 1 or len(xp) < 2 or not np.all(np.diff(xp) > 0.0):
        raise ValueError("xp must be a strictly increasing one-dimensional knot vector.")
    weights = np.zeros((len(x), len(xp)), dtype=dtype)
    for i, value in enumerate(x):
        if value <= xp[0]:
            left, right = 0, 1
        elif value >= xp[-1]:
            left, right = len(xp) - 2, len(xp) - 1
        else:
            right = int(np.searchsorted(xp, value, side="right"))
            left = right - 1
        t = (value - xp[left]) / (xp[right] - xp[left])
        weights[i, left] = 1.0 - t
        weights[i, right] = t
    return weights


def alpha_tau(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    return np.where(tau <= 30.0 / CALENDAR_DAYS_PER_YEAR, 0.8, 0.6)


def bsm_call_engine(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    dtype=np.float64,
) -> Dict[str, np.ndarray]:
    S = np.asarray(S, dtype=dtype)
    K = np.asarray(K, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    tau = np.maximum(np.asarray(tau, dtype=dtype), np.asarray(1e-10, dtype=dtype))
    sigma = np.maximum(np.asarray(sigma, dtype=dtype), np.asarray(1e-10, dtype=dtype))
    tiny = np.asarray(1e-14, dtype=dtype)
    sqrt_tau = np.sqrt(tau)
    log_ratio = np.log(np.maximum(S, tiny)) - np.log(np.maximum(K, tiny))
    d1 = (log_ratio + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    nd1 = norm_pdf(d1, dtype=dtype)
    w1 = np.asarray(ndtr(d1), dtype=dtype)
    w2 = np.asarray(ndtr(d2), dtype=dtype)
    discount = safe_exp(-r * tau, dtype=dtype)
    strike_ratio = K / np.maximum(S, tiny)
    strike_factor = strike_ratio * discount
    price_norm = w1 - strike_factor * w2
    price = S * price_norm
    delta = w1
    gamma = nd1 / np.maximum(S * sigma * sqrt_tau, tiny)
    spot_gamma_norm = S * gamma
    vega_norm = nd1 * sqrt_tau
    vega = S * vega_norm
    vanna = -nd1 * d2 / sigma
    volga_norm = vega_norm * d1 * d2 / sigma
    volga = S * volga_norm
    return {
        "price": price,
        "price_norm": price_norm,
        "delta": delta,
        "gamma": gamma,
        "spot_gamma_norm": spot_gamma_norm,
        "vega": vega,
        "vega_norm": vega_norm,
        "vanna": vanna,
        "volga": volga,
        "volga_norm": volga_norm,
        "w1": w1,
        "w2": w2,
        "strike_factor": strike_factor,
        "discount": discount,
        "d1": d1,
        "d2": d2,
    }


def normalized_call_value(k: np.ndarray, tau: np.ndarray, r: np.ndarray, sigma: np.ndarray, dtype=np.float64) -> np.ndarray:
    k = np.asarray(k, dtype=dtype)
    tau = np.asarray(tau, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    sigma = np.asarray(sigma, dtype=dtype)
    S = np.ones_like(k)
    K = safe_exp(k, dtype=dtype)
    return bsm_call_engine(S, K, r, tau, sigma, dtype=dtype)["price_norm"]


def build_teacher_panel(daily: pd.DataFrame, dtype=np.float64) -> pd.DataFrame:
    n_dates = len(daily)
    block = len(TAU_GRID_YEARS) * len(K_GRID)
    date_rep = np.repeat(daily["Date"].values, block)
    S_rep = np.repeat(daily["SPX_Close"].values, block)
    rv20_rep = np.repeat(daily["RV20"].values, block)
    rv60_rep = np.repeat(daily["RV60"].values, block)
    iv30_rep = np.repeat(daily["IV30"].values, block)
    iv90_rep = np.repeat(daily["IV90"].values, block)
    tau_rep = np.tile(np.repeat(TAU_GRID_YEARS, len(K_GRID)), n_dates)
    k_rep = np.tile(np.tile(K_GRID, len(TAU_GRID_YEARS)), n_dates)

    tau_block = np.repeat(TAU_GRID_YEARS, len(K_GRID))
    rate_x = np.array([0.25, 2.0, 10.0], dtype=dtype)
    iv_x = np.array([30.0 / CALENDAR_DAYS_PER_YEAR, 90.0 / CALENDAR_DAYS_PER_YEAR], dtype=dtype)

    # Fixed maturity grids permit a matrix formulation.  Interpolation weights
    # are built once and applied to every date in a single BLAS operation.
    rate_weights = piecewise_linear_weight_matrix(TAU_GRID_YEARS, rate_x, dtype=dtype)
    iv_weights = piecewise_linear_weight_matrix(TAU_GRID_YEARS, iv_x, dtype=dtype)
    rate_nodes = daily[["DGS3MO", "DGS2", "DGS10"]].to_numpy(dtype=dtype) / np.asarray(100.0, dtype=dtype)
    iv_nodes = daily[["IV30", "IV90"]].to_numpy(dtype=dtype)
    rates_by_date_tau = rate_nodes @ rate_weights.T
    iv_by_date_tau = np.maximum(iv_nodes @ iv_weights.T, np.asarray(1e-8, dtype=dtype))
    rates = np.repeat(rates_by_date_tau, len(K_GRID), axis=1).reshape(-1)
    iv_tau = np.repeat(iv_by_date_tau, len(K_GRID), axis=1).reshape(-1)

    sigma = alpha_tau(tau_rep) * iv_tau + (1.0 - alpha_tau(tau_rep)) * rv20_rep
    K_rep = S_rep * safe_exp(k_rep, dtype=dtype)
    labels = bsm_call_engine(S_rep, K_rep, rates, tau_rep, sigma, dtype=dtype)

    panel = pd.DataFrame({
        "date": pd.to_datetime(date_rep),
        "S": S_rep,
        "K": K_rep,
        "k": k_rep,
        "tau": tau_rep,
        "r_tau": rates,
        "RV20": rv20_rep,
        "RV60": rv60_rep,
        "IV30": iv30_rep,
        "IV90": iv90_rep,
        "IV_slope": iv90_rep - iv30_rep,
        "IV_tau": iv_tau,
        "sigma_teacher": sigma,
        "price_teacher": labels["price"],
        "price_teacher_normalized": labels["price_norm"],
        "delta_teacher": labels["delta"],
        "gamma_teacher": labels["gamma"],
        "spot_gamma_teacher_normalized": labels["spot_gamma_norm"],
        "vega_teacher": labels["vega"],
        "vega_teacher_normalized": labels["vega_norm"],
        "vanna_teacher": labels["vanna"],
        "volga_teacher": labels["volga"],
        "volga_teacher_normalized": labels["volga_norm"],
        "weight_asset_teacher": labels["w1"],
        "weight_strike_teacher": labels["w2"],
        "strike_factor_teacher": labels["strike_factor"],
        "d1_teacher": labels["d1"],
        "d2_teacher": labels["d2"],
    })
    panel["high_curvature_region"] = (
        (panel["tau"] <= HIGH_CURVATURE_TAU_MAX)
        & (panel["k"].abs() <= HIGH_CURVATURE_K_MAX)
    ).astype(int)
    return panel


def assign_splits(panel: pd.DataFrame, use_mock: bool) -> pd.DataFrame:
    out = panel.copy().sort_values(["date", "tau", "k"]).reset_index(drop=True)
    if use_mock:
        dates = pd.Index(sorted(out["date"].unique()))
        if len(dates) < 5:
            raise ValueError("Mock panel requires at least five unique dates.")
        train_cut = max(1, int(0.60 * len(dates)))
        val_cut = max(train_cut + 1, int(0.80 * len(dates)))
        mapping = {
            date: "train" if i < train_cut else ("val" if i < val_cut else "test")
            for i, date in enumerate(dates)
        }
        out["split"] = out["date"].map(mapping)
    else:
        train_end = pd.Timestamp("2017-12-31")
        val_end = pd.Timestamp("2021-12-31")
        out["split"] = np.where(
            out["date"] <= train_end,
            "train",
            np.where(out["date"] <= val_end, "val", "test"),
        )
    counts = out.groupby("split").size().to_dict()
    if any(counts.get(name, 0) == 0 for name in ["train", "val", "test"]):
        raise ValueError("At least one chronological split is empty.")
    ranges = out.groupby("split")["date"].agg(["min", "max"])
    if not (ranges.loc["train", "max"] < ranges.loc["val", "min"] <= ranges.loc["val", "max"] < ranges.loc["test", "min"]):
        raise ValueError("Chronological splits overlap or are not ordered.")
    return out


def maybe_subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    return (
        df.sample(n=max_rows, random_state=seed)
        .sort_values(["date", "tau", "k"])
        .reset_index(drop=True)
    )


def fourth_order_first(f, x: float, h: float) -> float:
    return float((-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h))


def fourth_order_second(f, x: float, h: float) -> float:
    return float((-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h)) / (12.0 * h * h))


def teacher_derivative_audit(panel: pd.DataFrame, n_checks: int, seed: int, dtype=np.float64) -> pd.DataFrame:
    if n_checks <= 0:
        return pd.DataFrame()
    sample = panel.sample(n=min(n_checks, len(panel)), random_state=seed).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for idx, row in sample.iterrows():
        k = float(row["k"])
        tau = float(row["tau"])
        r = float(row["r_tau"])
        sigma = float(row["sigma_teacher"])
        hk1 = finite_difference_first_step(k, dtype=dtype)
        hk2 = finite_difference_second_step(k, dtype=dtype)
        hs1 = finite_difference_first_step(sigma, dtype=dtype)
        hs2 = finite_difference_second_step(sigma, dtype=dtype)
        hmix_k = finite_difference_mixed_step(k, dtype=dtype)
        hmix_s = finite_difference_mixed_step(sigma, dtype=dtype)

        fk = lambda kval: float(normalized_call_value(np.array([kval]), np.array([tau]), np.array([r]), np.array([sigma]), dtype=dtype)[0])
        fs = lambda sval: float(normalized_call_value(np.array([k]), np.array([tau]), np.array([r]), np.array([sval]), dtype=dtype)[0])
        c = fk(k)
        ck = fourth_order_first(fk, k, hk1)
        ckk = fourth_order_second(fk, k, hk2)
        cs = fourth_order_first(fs, sigma, hs1)
        css = fourth_order_second(fs, sigma, hs2)
        cks = (
            normalized_call_value(np.array([k + hmix_k]), np.array([tau]), np.array([r]), np.array([sigma + hmix_s]), dtype=dtype)[0]
            - normalized_call_value(np.array([k + hmix_k]), np.array([tau]), np.array([r]), np.array([sigma - hmix_s]), dtype=dtype)[0]
            - normalized_call_value(np.array([k - hmix_k]), np.array([tau]), np.array([r]), np.array([sigma + hmix_s]), dtype=dtype)[0]
            + normalized_call_value(np.array([k - hmix_k]), np.array([tau]), np.array([r]), np.array([sigma - hmix_s]), dtype=dtype)[0]
        ) / (4.0 * hmix_k * hmix_s)

        fd_values = {
            "price_norm": c,
            "delta": c - ck,
            "spot_gamma": ckk - ck,
            "vega_norm": cs,
            "vanna": cs - cks,
            "volga_norm": css,
        }
        teacher_values = {
            "price_norm": float(row["price_teacher_normalized"]),
            "delta": float(row["delta_teacher"]),
            "spot_gamma": float(row["spot_gamma_teacher_normalized"]),
            "vega_norm": float(row["vega_teacher_normalized"]),
            "vanna": float(row["vanna_teacher"]),
            "volga_norm": float(row["volga_teacher_normalized"]),
        }
        for quantity in fd_values:
            rows.append({
                "sample": idx,
                "quantity": quantity,
                "finite_difference_value": fd_values[quantity],
                "analytic_value": teacher_values[quantity],
                "absolute_error": abs(fd_values[quantity] - teacher_values[quantity]),
                "relative_error": abs(fd_values[quantity] - teacher_values[quantity]) / max(1e-12, abs(teacher_values[quantity])),
            })
    return pd.DataFrame(rows)


def data_validation_table(daily: pd.DataFrame, panel: pd.DataFrame, source_audit: pd.DataFrame) -> pd.DataFrame:
    expected = len(TAU_GRID_YEARS) * len(K_GRID)
    weights_recon = panel["weight_asset_teacher"] - panel["strike_factor_teacher"] * panel["weight_strike_teacher"]
    vega_gamma = panel["S"] * panel["sigma_teacher"] * panel["tau"] * panel["gamma_teacher"]
    checks = [
        ("daily_dates_unique", not daily["Date"].duplicated().any(), int(daily["Date"].duplicated().sum())),
        ("daily_dates_sorted", bool(daily["Date"].is_monotonic_increasing), "increasing"),
        ("source_age_limit_respected", bool((source_audit["rows_over_age_limit"] == 0).all()), int(source_audit["rows_over_age_limit"].sum())),
        ("panel_size_exact", len(panel) == len(daily) * expected, f"{len(panel)} vs {len(daily) * expected}"),
        ("contracts_per_date_exact", bool((panel.groupby("date").size() == expected).all()), expected),
        ("positive_state", bool((panel[["S", "K", "tau", "sigma_teacher"]] > 0.0).all().all()), "S,K,tau,sigma > 0"),
        ("price_scale_identity", bool(np.allclose(panel["price_teacher"], panel["S"] * panel["price_teacher_normalized"], rtol=1e-12, atol=1e-10)), "C=S c"),
        ("two_leg_identity", bool(np.allclose(weights_recon, panel["price_teacher_normalized"], rtol=1e-12, atol=1e-12)), float(np.max(np.abs(weights_recon - panel["price_teacher_normalized"])))),
        ("delta_asset_weight_identity", bool(np.allclose(panel["delta_teacher"], panel["weight_asset_teacher"], rtol=1e-12, atol=1e-12)), float(np.max(np.abs(panel["delta_teacher"] - panel["weight_asset_teacher"])))),
        ("vega_gamma_identity", bool(np.allclose(vega_gamma, panel["vega_teacher_normalized"], rtol=1e-11, atol=1e-12)), float(np.max(np.abs(vega_gamma - panel["vega_teacher_normalized"])))),
        ("finite_targets", bool(np.isfinite(panel[TARGET_COLS + ["weight_asset_teacher", "weight_strike_teacher"]].values).all()), "all finite"),
    ]
    return pd.DataFrame([{"check": name, "passed": bool(passed), "detail": str(detail)} for name, passed, detail in checks])


# ---------------------------------------------------------------------------
# Models, automatic derivatives, and training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    hidden_dim: int
    depth: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    gradient_clip: float
    lr_factor: float
    lr_patience: int
    consistency_fraction: float
    lambda_delta: float
    lambda_gamma: float
    lambda_vega: float
    lambda_vanna: float
    lambda_volga: float
    lambda_potential: float
    lambda_decomposition: float
    lambda_weights: float


class SmoothSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, model_kind: str):
        super().__init__()
        if model_kind not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unsupported model kind {model_kind}.")
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.SiLU()])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.model_kind = model_kind
        output_dim = 1 if model_kind == "A" else (8 if model_kind == "D" else 6)
        self.head = nn.Linear(in_dim, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.trunk(x))


@dataclass
class FittedSurrogate:
    name: str
    kind: str
    model: SmoothSurrogate
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    config: TrainConfig
    seed: int
    best_epoch: int
    best_validation_score: float
    dtype_name: str
    feature_cols: List[str]
    target_cols: List[str]


@dataclass
class PotentialBundle:
    c: Tensor
    delta: Tensor
    spot_gamma: Tensor
    vega_norm: Tensor
    vanna: Tensor
    volga_norm: Tensor


@dataclass
class PreparedSeedData:
    """Reusable standardized arrays/tensors shared by all four models for a seed."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_x_cpu: Tensor
    train_y_cpu: Tensor
    train_w_cpu: Tensor
    val_x_std: np.ndarray
    test_x_std: np.ndarray
    train_x_device: Optional[Tensor] = None
    train_y_device: Optional[Tensor] = None
    train_w_device: Optional[Tensor] = None


@dataclass
class HedgingStateCache:
    states: pd.DataFrame
    n_paths: int
    spot: np.ndarray
    rate: np.ndarray
    dt: np.ndarray
    teacher_final: np.ndarray


def scaler_tensors(scaler: StandardScaler, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    mean = torch.as_tensor(scaler.mean_, device=device, dtype=dtype)
    scale = torch.as_tensor(scaler.scale_, device=device, dtype=dtype)
    return mean, scale


def raw_base_heads(output: Tensor, y_mean: Tensor, y_scale: Tensor) -> Tensor:
    base = output[:, : min(6, output.shape[1])]
    return base * y_scale[: base.shape[1]] + y_mean[: base.shape[1]]


def reconstructed_price_from_output(
    output: Tensor,
    x_std: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    if output.shape[1] < 8:
        return output[:, 0], None
    x_raw = x_std * x_scale + x_mean
    k = x_raw[:, FEATURE_COLS.index("k")]
    tau = x_raw[:, FEATURE_COLS.index("tau")]
    r = x_raw[:, FEATURE_COLS.index("r_tau")]
    strike_factor = torch.exp(torch.clamp(k - r * tau, min=-50.0, max=50.0))
    weights = torch.sigmoid(output[:, 6:8])
    c_dec_std_placeholder = weights[:, 0] - strike_factor * weights[:, 1]
    return c_dec_std_placeholder, weights


def price_potential_from_output(
    model_kind: str,
    output: Tensor,
    x_std: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    direct_raw = raw_base_heads(output, y_mean, y_scale)
    if model_kind == "D":
        x_raw = x_std * x_scale + x_mean
        k = x_raw[:, FEATURE_COLS.index("k")]
        tau = x_raw[:, FEATURE_COLS.index("tau")]
        r = x_raw[:, FEATURE_COLS.index("r_tau")]
        strike_factor = torch.exp(torch.clamp(k - r * tau, min=-50.0, max=50.0))
        weights = torch.sigmoid(output[:, 6:8])
        c = weights[:, 0] - strike_factor * weights[:, 1]
    else:
        weights = None
        c = direct_raw[:, 0]
    return c, direct_raw, weights


def price_potential_raw(
    model: SmoothSurrogate,
    x_std: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    output = model(x_std)
    return price_potential_from_output(
        model.model_kind, output, x_std, x_mean, x_scale, y_mean, y_scale
    )


def potential_derivatives(
    model: SmoothSurrogate,
    x_std: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
    *,
    create_graph: bool,
) -> Tuple[PotentialBundle, Tensor, Optional[Tensor]]:
    if not x_std.requires_grad:
        x_std = x_std.detach().clone().requires_grad_(True)
    output = model(x_std)
    return potential_derivatives_from_output(
        model.model_kind,
        output,
        x_std,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        create_graph=create_graph,
    )


def potential_derivatives_from_output(
    model_kind: str,
    output: Tensor,
    x_std: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
    *,
    create_graph: bool,
) -> Tuple[PotentialBundle, Tensor, Optional[Tensor]]:
    """Differentiate a price potential using an already-computed forward pass."""
    if not x_std.requires_grad:
        raise ValueError("x_std must require gradients when reusing a forward pass.")
    c, direct_raw, weights = price_potential_from_output(
        model_kind, output, x_std, x_mean, x_scale, y_mean, y_scale
    )
    grad_c_std = torch.autograd.grad(c.sum(), x_std, create_graph=True, retain_graph=True)[0]
    k_idx = FEATURE_COLS.index("k")
    sigma_idx = FEATURE_COLS.index("sigma_teacher")
    c_k = grad_c_std[:, k_idx] / x_scale[k_idx]
    c_sigma = grad_c_std[:, sigma_idx] / x_scale[sigma_idx]
    grad_ck_std = torch.autograd.grad(c_k.sum(), x_std, create_graph=True, retain_graph=True)[0]
    grad_cs_std = torch.autograd.grad(c_sigma.sum(), x_std, create_graph=True, retain_graph=True)[0]
    c_kk = grad_ck_std[:, k_idx] / x_scale[k_idx]
    c_ks = grad_ck_std[:, sigma_idx] / x_scale[sigma_idx]
    c_ss = grad_cs_std[:, sigma_idx] / x_scale[sigma_idx]
    bundle = PotentialBundle(
        c=c,
        delta=c - c_k,
        spot_gamma=c_kk - c_k,
        vega_norm=c_sigma,
        vanna=c_sigma - c_ks,
        volga_norm=c_ss,
    )
    if not create_graph:
        bundle = PotentialBundle(*(v.detach() for v in bundle.__dict__.values()))
        direct_raw = direct_raw.detach()
        weights = None if weights is None else weights.detach()
    return bundle, direct_raw, weights


def make_tensor_dataset(df: pd.DataFrame, x_scaler: StandardScaler, y_scaler: StandardScaler, dtype: torch.dtype) -> TensorDataset:
    x = x_scaler.transform(df[FEATURE_COLS].values.astype(float))
    y = y_scaler.transform(df[TARGET_COLS].values.astype(float))
    weights = df[["weight_asset_teacher", "weight_strike_teacher"]].values.astype(float)
    return TensorDataset(
        torch.as_tensor(x, dtype=dtype),
        torch.as_tensor(y, dtype=dtype),
        torch.as_tensor(weights, dtype=dtype),
    )


def prepare_seed_data(
    panel: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedSeedData:
    """Prepare split frames, scalers, and standardized tensors once per seed.

    Earlier implementations repeated this work for Models A--D. The prepared
    object is immutable in ordinary use and can be shared by all model fits
    and evaluations.
    """
    train_df = maybe_subsample(panel[panel["split"] == "train"], args.max_train_samples, seed)
    val_df = maybe_subsample(panel[panel["split"] == "val"], args.max_val_samples, seed + 1)
    test_df = maybe_subsample(panel[panel["split"] == "test"], args.max_test_samples, seed + 2)
    x_scaler = StandardScaler().fit(train_df[FEATURE_COLS].values.astype(float))
    y_scaler = StandardScaler().fit(train_df[TARGET_COLS].values.astype(float))

    train_x_np = x_scaler.transform(train_df[FEATURE_COLS].values.astype(float))
    train_y_np = y_scaler.transform(train_df[TARGET_COLS].values.astype(float))
    train_w_np = train_df[["weight_asset_teacher", "weight_strike_teacher"]].values.astype(float)
    val_x_std = x_scaler.transform(val_df[FEATURE_COLS].values.astype(float))
    test_x_std = x_scaler.transform(test_df[FEATURE_COLS].values.astype(float))

    pin = bool(args.pin_memory and device.type == "cuda")
    train_x_cpu = torch.as_tensor(train_x_np, dtype=dtype)
    train_y_cpu = torch.as_tensor(train_y_np, dtype=dtype)
    train_w_cpu = torch.as_tensor(train_w_np, dtype=dtype)
    if pin:
        train_x_cpu = train_x_cpu.pin_memory()
        train_y_cpu = train_y_cpu.pin_memory()
        train_w_cpu = train_w_cpu.pin_memory()

    prepared = PreparedSeedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_x_cpu=train_x_cpu,
        train_y_cpu=train_y_cpu,
        train_w_cpu=train_w_cpu,
        val_x_std=val_x_std,
        test_x_std=test_x_std,
    )

    bytes_required = sum(t.numel() * t.element_size() for t in [train_x_cpu, train_y_cpu, train_w_cpu])
    preload_limit = int(args.max_preload_gb * (1024 ** 3))
    if args.preload_training_data and device.type == "cuda" and bytes_required <= preload_limit:
        prepared.train_x_device = train_x_cpu.to(device=device, non_blocking=pin)
        prepared.train_y_device = train_y_cpu.to(device=device, non_blocking=pin)
        prepared.train_w_device = train_w_cpu.to(device=device, non_blocking=pin)
    return prepared


def supervised_head_weights(config: TrainConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [1.0, config.lambda_delta, config.lambda_gamma, config.lambda_vega, config.lambda_vanna, config.lambda_volga],
        device=device,
        dtype=dtype,
    )


def sample_consistency_indices(n: int, fraction: float, device: torch.device) -> Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("consistency_fraction must lie in (0,1].")
    m = max(1, int(math.ceil(fraction * n)))
    if m >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:m]


def supervised_loss_sums(
    model_kind: str,
    output: Tensor,
    x_std: Tensor,
    y_std: Tensor,
    weight_teacher: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
    head_weights: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return unnormalised base, decomposition, and digital-weight losses."""
    if model_kind == "A":
        base_sum = torch.sum((output[:, 0] - y_std[:, 0]) ** 2)
        zero = torch.zeros((), device=output.device, dtype=output.dtype)
        return base_sum, zero, zero

    direct_std = output[:, :6]
    decomposition_sum = torch.zeros((), device=output.device, dtype=output.dtype)
    weight_sum = torch.zeros_like(decomposition_sum)
    if model_kind == "D":
        x_raw = x_std * x_scale + x_mean
        k = x_raw[:, FEATURE_COLS.index("k")]
        tau = x_raw[:, FEATURE_COLS.index("tau")]
        r = x_raw[:, FEATURE_COLS.index("r_tau")]
        strike_factor = torch.exp(torch.clamp(k - r * tau, min=-50.0, max=50.0))
        weight_pred = torch.sigmoid(output[:, 6:8])
        c_dec = weight_pred[:, 0] - strike_factor * weight_pred[:, 1]
        c_dec_std = (c_dec - y_mean[0]) / y_scale[0]
        supervised_std = torch.cat([c_dec_std[:, None], direct_std[:, 1:6]], dim=1)
        direct_c_raw = direct_std[:, 0] * y_scale[0] + y_mean[0]
        decomposition_sum = torch.sum(((direct_c_raw - c_dec) / y_scale[0]) ** 2)
        weight_sum = torch.sum((weight_pred - weight_teacher) ** 2)
    else:
        supervised_std = direct_std
    base_sum = torch.sum(head_weights * (supervised_std - y_std[:, :6]) ** 2)
    return base_sum, decomposition_sum, weight_sum


def compute_batch_loss(
    model: SmoothSurrogate,
    x_std: Tensor,
    y_std: Tensor,
    weight_teacher: Tensor,
    x_mean: Tensor,
    x_scale: Tensor,
    y_mean: Tensor,
    y_scale: Tensor,
    config: TrainConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    head_weights = supervised_head_weights(config, x_std.device, x_std.dtype)
    n = int(x_std.shape[0])
    potential_loss = torch.zeros((), device=x_std.device, dtype=x_std.dtype)

    if model.model_kind not in {"C", "D"}:
        output = model(x_std)
        base_sum, decomposition_sum, weight_sum = supervised_loss_sums(
            model.model_kind, output, x_std, y_std, weight_teacher,
            x_mean, x_scale, y_mean, y_scale, head_weights,
        )
    else:
        # Partition the batch.  Consistency rows receive one differentiable
        # forward pass that is reused for both supervised and potential losses;
        # non-consistency rows receive only the inexpensive supervised pass.
        idx_cons = sample_consistency_indices(n, config.consistency_fraction, x_std.device)
        mask_plain = torch.ones(n, dtype=torch.bool, device=x_std.device)
        mask_plain[idx_cons] = False
        base_sum = torch.zeros((), device=x_std.device, dtype=x_std.dtype)
        decomposition_sum = torch.zeros_like(base_sum)
        weight_sum = torch.zeros_like(base_sum)

        if bool(mask_plain.any()):
            x_plain = x_std[mask_plain]
            y_plain = y_std[mask_plain]
            w_plain = weight_teacher[mask_plain]
            output_plain = model(x_plain)
            b, d, w = supervised_loss_sums(
                model.model_kind, output_plain, x_plain, y_plain, w_plain,
                x_mean, x_scale, y_mean, y_scale, head_weights,
            )
            base_sum = base_sum + b
            decomposition_sum = decomposition_sum + d
            weight_sum = weight_sum + w

        x_cons = x_std[idx_cons].detach().clone().requires_grad_(True)
        y_cons = y_std[idx_cons]
        w_cons = weight_teacher[idx_cons]
        output_cons = model(x_cons)
        b, d, w = supervised_loss_sums(
            model.model_kind, output_cons, x_cons, y_cons, w_cons,
            x_mean, x_scale, y_mean, y_scale, head_weights,
        )
        base_sum = base_sum + b
        decomposition_sum = decomposition_sum + d
        weight_sum = weight_sum + w

        potential, direct_raw, _ = potential_derivatives_from_output(
            model.model_kind, output_cons, x_cons,
            x_mean, x_scale, y_mean, y_scale, create_graph=True,
        )
        potential_matrix = torch.stack([
            potential.delta,
            potential.spot_gamma,
            potential.vega_norm,
            potential.vanna,
            potential.volga_norm,
        ], dim=1)
        direct_matrix = direct_raw[:, 1:6]
        potential_loss = torch.mean(
            head_weights[1:6] * ((direct_matrix - potential_matrix) / y_scale[1:6]) ** 2
        )

    base_denominator = float(n if model.model_kind == "A" else n * 6)
    base_loss = base_sum / base_denominator
    decomposition_loss = decomposition_sum / max(n, 1)
    weight_loss = weight_sum / max(2 * n, 1)
    total = base_loss + config.lambda_potential * potential_loss
    if model.model_kind == "D":
        total = total + config.lambda_decomposition * decomposition_loss + config.lambda_weights * weight_loss

    return total, {
        "base": float(base_loss.detach()),
        "potential": float(potential_loss.detach()),
        "decomposition": float(decomposition_loss.detach()),
        "weights": float(weight_loss.detach()),
    }


def predict_normalized_targets(
    fitted: FittedSurrogate,
    df: pd.DataFrame,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    include_consistency: bool = True,
    x_standardized: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    model = fitted.model.to(device=device, dtype=dtype)
    model.eval()
    x_mean, x_scale = scaler_tensors(fitted.x_scaler, device, dtype)
    y_mean, y_scale = scaler_tensors(fitted.y_scaler, device, dtype)
    x_all = (
        fitted.x_scaler.transform(df[FEATURE_COLS].values.astype(float))
        if x_standardized is None
        else np.asarray(x_standardized, dtype=float)
    )
    if len(x_all) != len(df):
        raise ValueError("x_standardized length does not match the prediction frame.")

    pieces: List[pd.DataFrame] = []
    for start in range(0, len(df), batch_size):
        stop = min(len(df), start + batch_size)
        x_base = torch.as_tensor(x_all[start:stop], device=device, dtype=dtype)

        need_potential = fitted.kind == "A" or include_consistency
        if need_potential:
            x = x_base.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                potential, direct_raw, weight_pred = potential_derivatives(
                    model, x, x_mean, x_scale, y_mean, y_scale, create_graph=False
                )
        else:
            with torch.no_grad():
                output = model(x_base)
                direct_raw = raw_base_heads(output, y_mean, y_scale)
                potential = None
                weight_pred = None
                if fitted.kind == "D":
                    x_raw = x_base * x_scale + x_mean
                    k = x_raw[:, FEATURE_COLS.index("k")]
                    tau = x_raw[:, FEATURE_COLS.index("tau")]
                    r = x_raw[:, FEATURE_COLS.index("r_tau")]
                    strike_factor = torch.exp(torch.clamp(k - r * tau, min=-50.0, max=50.0))
                    weight_pred = torch.sigmoid(output[:, 6:8])
                    c_dec = weight_pred[:, 0] - strike_factor * weight_pred[:, 1]

        if fitted.kind == "A":
            assert potential is not None
            values = torch.stack([
                potential.c,
                potential.delta,
                potential.spot_gamma,
                potential.vega_norm,
                potential.vanna,
                potential.volga_norm,
            ], dim=1)
            direct_for_gap = None
        elif fitted.kind == "D":
            c_value = potential.c if potential is not None else c_dec
            values = torch.stack([
                c_value,
                direct_raw[:, 1],
                direct_raw[:, 2],
                direct_raw[:, 3],
                direct_raw[:, 4],
                direct_raw[:, 5],
            ], dim=1)
            direct_for_gap = direct_raw
        else:
            values = direct_raw[:, :6]
            direct_for_gap = direct_raw

        frag = pd.DataFrame(values.detach().cpu().numpy(), columns=[f"pred_{x}" for x in TARGET_LABELS])
        if include_consistency and direct_for_gap is not None:
            assert potential is not None
            potential_matrix = torch.stack([
                potential.delta,
                potential.spot_gamma,
                potential.vega_norm,
                potential.vanna,
                potential.volga_norm,
            ], dim=1)
            gap = direct_for_gap[:, 1:6] - potential_matrix
            for j, label in enumerate(TARGET_LABELS[1:]):
                frag[f"gap_{label}"] = gap[:, j].detach().cpu().numpy()
        if fitted.kind == "D" and weight_pred is not None:
            frag["pred_weight_asset"] = weight_pred[:, 0].detach().cpu().numpy()
            frag["pred_weight_strike"] = weight_pred[:, 1].detach().cpu().numpy()
            direct_c = direct_raw[:, 0]
            c_value = potential.c if potential is not None else c_dec
            frag["decomposition_gap"] = (direct_c - c_value).detach().cpu().numpy()
        pieces.append(frag)
    return pd.concat(pieces, ignore_index=True)


def validation_score_from_predictions(pred: pd.DataFrame, df: pd.DataFrame, y_scaler: StandardScaler) -> float:
    score = 0.0
    weights = np.array([1.0, 1.0, 1.0, 0.75, 0.75, 0.75], dtype=float)
    for j, label in enumerate(TARGET_LABELS):
        teacher = df[TARGET_COLS[j]].values.astype(float)
        predicted = pred[f"pred_{label}"].values.astype(float)
        score += weights[j] * rmse(predicted, teacher) / max(float(y_scaler.scale_[j]), 1e-12)
    return float(score / weights.sum())


def fit_one_model(
    name: str,
    kind: str,
    prepared: PreparedSeedData,
    seed: int,
    config: TrainConfig,
    dtype_name: str,
    device: torch.device,
    deterministic: bool,
    *,
    validation_interval: int = 1,
    log_every: int = 10,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 5,
    resume_checkpoint: bool = True,
    run_progress: Optional[RunProgress] = None,
) -> Tuple[FittedSurrogate, pd.DataFrame]:
    """Fit one model using data/scalers cached once for the matched seed.

    Training tensors can remain resident on the GPU.  A compact checkpoint is
    written periodically and can resume an interrupted model without repeating
    completed epochs.
    """
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive.")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")

    set_seed(seed, deterministic=deterministic)
    dtype = resolve_torch_dtype(dtype_name)
    x_scaler = prepared.x_scaler
    y_scaler = prepared.y_scaler
    model = SmoothSurrogate(len(FEATURE_COLS), config.hidden_dim, config.depth, kind).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=max(1, int(math.ceil(config.lr_patience / validation_interval))),
        min_lr=1e-7,
    )
    x_mean, x_scale = scaler_tensors(x_scaler, device, dtype)
    y_mean, y_scale = scaler_tensors(y_scaler, device, dtype)

    use_device_cache = prepared.train_x_device is not None
    train_x = prepared.train_x_device if use_device_cache else prepared.train_x_cpu
    train_y = prepared.train_y_device if use_device_cache else prepared.train_y_cpu
    train_w = prepared.train_w_device if use_device_cache else prepared.train_w_cpu
    assert train_x is not None and train_y is not None and train_w is not None
    n_train = int(train_x.shape[0])
    batch_size = min(config.batch_size, n_train)
    pin = bool(device.type == "cuda" and prepared.train_x_cpu.is_pinned())

    best_state: Optional[Dict[str, Tensor]] = None
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history_rows: List[Dict[str, Any]] = []
    start_epoch = 1
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed)

    if checkpoint_path is not None and resume_checkpoint and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            expected = {
                "name": name,
                "kind": kind,
                "seed": seed,
                "dtype_name": dtype_name,
                "config": asdict(config),
            }
            if all(checkpoint.get(k) == v for k, v in expected.items()):
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                for state in optimizer.state.values():
                    for key, value in list(state.items()):
                        if torch.is_tensor(value):
                            state[key] = value.to(device=device)
                best_state = checkpoint.get("best_state")
                best_score = float(checkpoint.get("best_score", float("inf")))
                best_epoch = int(checkpoint.get("best_epoch", 0))
                stale_epochs = int(checkpoint.get("stale_epochs", 0))
                history_rows = list(checkpoint.get("history_rows", []))
                start_epoch = int(checkpoint["epoch"]) + 1
                generator_state = checkpoint.get("permutation_generator_state")
                if generator_state is not None:
                    permutation_generator.set_state(generator_state)
                if checkpoint.get("python_random_state") is not None:
                    random.setstate(checkpoint["python_random_state"])
                if checkpoint.get("numpy_random_state") is not None:
                    np.random.set_state(checkpoint["numpy_random_state"])
                if checkpoint.get("torch_rng_state") is not None:
                    torch.set_rng_state(checkpoint["torch_rng_state"])
                if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state_all") is not None:
                    torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
                print(f"[{name} seed={seed}] resumed at epoch {start_epoch}/{config.max_epochs}", flush=True)
            else:
                warnings.warn(f"Ignoring incompatible checkpoint {checkpoint_path}.", RuntimeWarning)
        except Exception as exc:
            warnings.warn(f"Could not resume checkpoint {checkpoint_path}: {exc}", RuntimeWarning)

    model_start = time.perf_counter()
    last_validation_epoch = max(0, start_epoch - 1)
    for epoch in range(start_epoch, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = {"total": 0.0, "base": 0.0, "potential": 0.0, "decomposition": 0.0, "weights": 0.0}
        n_batches = 0
        permutation_cpu = torch.randperm(n_train, generator=permutation_generator, device="cpu")
        for start in range(0, n_train, batch_size):
            idx_cpu = permutation_cpu[start : start + batch_size]
            if use_device_cache:
                idx = idx_cpu.to(device=device, non_blocking=True)
                x = train_x[idx]
                y = train_y[idx]
                w = train_w[idx]
            else:
                x = train_x[idx_cpu].to(device=device, non_blocking=pin)
                y = train_y[idx_cpu].to(device=device, non_blocking=pin)
                w = train_w[idx_cpu].to(device=device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_batch_loss(model, x, y, w, x_mean, x_scale, y_mean, y_scale, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {name}, epoch {epoch}.")
            loss.backward()
            if config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running["total"] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
            n_batches += 1

        should_validate = epoch == 1 or epoch % validation_interval == 0 or epoch == config.max_epochs
        val_score = float("nan")
        if should_validate:
            fitted_temp = FittedSurrogate(
                name=name,
                kind=kind,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                config=config,
                seed=seed,
                best_epoch=epoch,
                best_validation_score=float("nan"),
                dtype_name=dtype_name,
                feature_cols=list(FEATURE_COLS),
                target_cols=list(TARGET_COLS),
            )
            pred_val = predict_normalized_targets(
                fitted_temp,
                prepared.val_df,
                device=device,
                dtype=dtype,
                batch_size=max(64, config.batch_size),
                include_consistency=False,
                x_standardized=prepared.val_x_std,
            )
            val_score = validation_score_from_predictions(pred_val, prepared.val_df, y_scaler)
            scheduler.step(val_score)
            elapsed_since_validation = max(1, epoch - last_validation_epoch)
            last_validation_epoch = epoch
            if val_score < best_score - config.min_delta:
                best_score = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += elapsed_since_validation

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append({
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": running["total"] / max(n_batches, 1),
            "train_base_loss": running["base"] / max(n_batches, 1),
            "train_potential_loss": running["potential"] / max(n_batches, 1),
            "train_decomposition_loss": running["decomposition"] / max(n_batches, 1),
            "train_weight_loss": running["weights"] / max(n_batches, 1),
            "validation_score": val_score,
            "learning_rate": current_lr,
            "epoch_seconds": time.perf_counter() - epoch_start,
        })

        if checkpoint_path is not None and (epoch % checkpoint_every == 0 or should_validate):
            ensure_dir(checkpoint_path.parent)
            checkpoint_payload = {
                "name": name,
                "kind": kind,
                "seed": seed,
                "dtype_name": dtype_name,
                "config": asdict(config),
                "epoch": epoch,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history_rows": history_rows,
                "permutation_generator_state": permutation_generator.get_state(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(checkpoint_payload, checkpoint_path)

        early_stop_now = bool(should_validate and stale_epochs >= config.patience)
        should_log = (
            log_every > 0
            and (epoch == 1 or epoch % log_every == 0 or epoch == config.max_epochs or early_stop_now)
        )
        if should_log:
            val_text = "--" if not np.isfinite(val_score) else f"{val_score:.6g}"
            progress_text = ""
            if run_progress is not None:
                progress_text = " " + run_progress.epoch_suffix(
                    kind, epoch, config.max_epochs, time.perf_counter() - model_start
                )
            print(
                f"[{name} seed={seed}] epoch={epoch}/{config.max_epochs} "
                f"loss={history_rows[-1]['train_total_loss']:.6g} "
                f"val={val_text} best={best_score:.6g} stale={stale_epochs}/{config.patience} "
                f"lr={current_lr:.3g} time={history_rows[-1]['epoch_seconds']:.1f}s"
                f"{progress_text}",
                flush=True,
            )
        if early_stop_now:
            print(
                f"[{name} seed={seed}] early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch} patience={config.patience}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was produced for {name}.")
    model.load_state_dict(best_state)
    model.to(device=device, dtype=dtype)
    fitted = FittedSurrogate(
        name=name,
        kind=kind,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        config=config,
        seed=seed,
        best_epoch=best_epoch,
        best_validation_score=best_score,
        dtype_name=dtype_name,
        feature_cols=list(FEATURE_COLS),
        target_cols=list(TARGET_COLS),
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
    final_epoch = int(history_rows[-1]["epoch"]) if history_rows else best_epoch
    stop_reason = "max_epochs" if final_epoch >= config.max_epochs else "early_stopping"
    print(
        f"[{name} seed={seed}] completed final_epoch={final_epoch} best_epoch={best_epoch} "
        f"best_val={best_score:.6g} stop={stop_reason} "
        f"elapsed={(time.perf_counter() - model_start) / 60.0:.1f} min",
        flush=True,
    )
    return fitted, pd.DataFrame(history_rows)


def prediction_frame(
    fitted: FittedSurrogate,
    df: pd.DataFrame,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    x_standardized: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    normalized = predict_normalized_targets(
        fitted,
        df,
        device,
        dtype,
        batch_size,
        include_consistency=True,
        x_standardized=x_standardized,
    )
    out = df[[
        "date", "S", "K", "k", "tau", "r_tau", "sigma_teacher",
        "price_teacher", "price_teacher_normalized", "delta_teacher", "gamma_teacher",
        "spot_gamma_teacher_normalized", "vega_teacher", "vega_teacher_normalized",
        "vanna_teacher", "volga_teacher", "volga_teacher_normalized",
        "weight_asset_teacher", "weight_strike_teacher", "high_curvature_region",
    ]].copy().reset_index(drop=True)
    suffix = fitted.name
    out[f"price_normalized_{suffix}"] = normalized["pred_price_norm"].values
    out[f"price_{suffix}"] = out["S"].values * normalized["pred_price_norm"].values
    out[f"delta_{suffix}"] = normalized["pred_delta"].values
    out[f"spot_gamma_normalized_{suffix}"] = normalized["pred_spot_gamma"].values
    out[f"gamma_{suffix}"] = normalized["pred_spot_gamma"].values / np.maximum(out["S"].values, 1e-12)
    out[f"vega_normalized_{suffix}"] = normalized["pred_vega_norm"].values
    out[f"vega_{suffix}"] = out["S"].values * normalized["pred_vega_norm"].values
    out[f"vanna_{suffix}"] = normalized["pred_vanna"].values
    out[f"volga_normalized_{suffix}"] = normalized["pred_volga_norm"].values
    out[f"volga_{suffix}"] = out["S"].values * normalized["pred_volga_norm"].values
    for column in normalized.columns:
        if column.startswith("gap_") or column in {"pred_weight_asset", "pred_weight_strike", "decomposition_gap"}:
            out[f"{column}_{suffix}"] = normalized[column].values
    return out


# ---------------------------------------------------------------------------
# Evaluation, inference, and hedging
# ---------------------------------------------------------------------------

def metric_rows_for_frame(frame: pd.DataFrame, model_name: str, region: str) -> List[Dict[str, Any]]:
    if region == "global":
        frag = frame
    elif region == "high_curvature":
        frag = frame[frame["high_curvature_region"] == 1]
    else:
        raise ValueError(region)
    mapping = {
        "price": (f"price_{model_name}", "price_teacher"),
        "delta": (f"delta_{model_name}", "delta_teacher"),
        "gamma": (f"gamma_{model_name}", "gamma_teacher"),
        "vega": (f"vega_{model_name}", "vega_teacher"),
        "vanna": (f"vanna_{model_name}", "vanna_teacher"),
        "volga": (f"volga_{model_name}", "volga_teacher"),
    }
    return [{
        "model": model_name,
        "region": region,
        "quantity": quantity,
        "rmse": rmse(frag[pred].values, frag[teacher].values),
        "mae": mae(frag[pred].values, frag[teacher].values),
        "n_samples": len(frag),
    } for quantity, (pred, teacher) in mapping.items()]


def consistency_metrics(frame: pd.DataFrame, model_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label in TARGET_LABELS[1:]:
        col = f"gap_{label}_{model_name}"
        if col in frame.columns:
            rows.append({
                "model": model_name,
                "consistency_quantity": label,
                "gap_rmse": rmse(frame[col].values, np.zeros(len(frame))),
                "gap_mae": mae(frame[col].values, np.zeros(len(frame))),
            })
    if model_name == "Model_D":
        if f"decomposition_gap_{model_name}" in frame.columns:
            rows.append({
                "model": model_name,
                "consistency_quantity": "price_decomposition",
                "gap_rmse": rmse(frame[f"decomposition_gap_{model_name}"].values, np.zeros(len(frame))),
                "gap_mae": mae(frame[f"decomposition_gap_{model_name}"].values, np.zeros(len(frame))),
            })
        for leg in ["asset", "strike"]:
            pred = f"pred_weight_{leg}_{model_name}"
            teacher = f"weight_{leg}_teacher"
            if pred in frame.columns:
                rows.append({
                    "model": model_name,
                    "consistency_quantity": f"{leg}_weight",
                    "gap_rmse": rmse(frame[pred].values, frame[teacher].values),
                    "gap_mae": mae(frame[pred].values, frame[teacher].values),
                })
    return rows


def paired_inference(values: np.ndarray, confidence_level: float) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n_seeds": 0, "mean_improvement": np.nan, "sd_improvement": np.nan, "standard_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value_two_sided": np.nan, "win_rate": np.nan}
    mean = float(np.mean(x))
    win_rate = float(np.mean(x > 0.0))
    if n == 1:
        return {"n_seeds": 1, "mean_improvement": mean, "sd_improvement": np.nan, "standard_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value_two_sided": np.nan, "win_rate": win_rate}
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(student_t_dist.ppf(0.5 + confidence_level / 2.0, df=n - 1))
    p_value = float(ttest_1samp(x, popmean=0.0).pvalue) if sd > 0.0 else (0.0 if mean != 0.0 else 1.0)
    return {
        "n_seeds": n,
        "mean_improvement": mean,
        "sd_improvement": sd,
        "standard_error": se,
        "ci_lower": mean - critical * se,
        "ci_upper": mean + critical * se,
        "p_value_two_sided": p_value,
        "win_rate": win_rate,
    }


def paired_metric_summary(per_seed_metrics: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    piv = per_seed_metrics.pivot_table(index=["seed", "region", "quantity"], columns="model", values="rmse").reset_index()
    comparisons = [("Model_A", "Model_B"), ("Model_B", "Model_C"), ("Model_C", "Model_D"), ("Model_A", "Model_D")]
    rows: List[Dict[str, Any]] = []
    for region in piv["region"].unique():
        for quantity in piv["quantity"].unique():
            frag = piv[(piv["region"] == region) & (piv["quantity"] == quantity)]
            for baseline, treatment in comparisons:
                if baseline not in frag.columns or treatment not in frag.columns:
                    continue
                improvements = frag[baseline].values - frag[treatment].values
                infer = paired_inference(improvements, confidence_level)
                rows.append({
                    "region": region,
                    "quantity": quantity,
                    "baseline": baseline,
                    "treatment": treatment,
                    "baseline_mean_rmse": float(frag[baseline].mean()),
                    "treatment_mean_rmse": float(frag[treatment].mean()),
                    **infer,
                    "supported_at_confidence_level": bool(np.isfinite(infer["ci_lower"]) and infer["ci_lower"] > 0.0),
                    "confidence_level": confidence_level,
                })
    return pd.DataFrame(rows)


def daily_lookup(daily: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
    return {
        pd.Timestamp(row["Date"]): {
            "S": float(row["SPX_Close"]),
            "RV20": float(row["RV20"]),
            "RV60": float(row["RV60"]),
            "IV30": float(row["IV30"]),
            "IV90": float(row["IV90"]),
            "DGS3MO": float(row["DGS3MO"]) / 100.0,
            "DGS2": float(row["DGS2"]) / 100.0,
            "DGS10": float(row["DGS10"]) / 100.0,
        }
        for _, row in daily.iterrows()
    }


def remaining_tau(start: pd.Timestamp, current: pd.Timestamp, initial_tau: float) -> float:
    return max(initial_tau - (current - start).days / CALENDAR_DAYS_PER_YEAR, 0.0)


def contract_state(date: pd.Timestamp, K: float, tau: float, state: Dict[str, float]) -> Dict[str, float]:
    k = math.log(max(K, 1e-12) / max(state["S"], 1e-12))
    r = float(piecewise_linear_extrap(np.array([tau]), np.array([0.25, 2.0, 10.0]), np.array([state["DGS3MO"], state["DGS2"], state["DGS10"]]))[0])
    iv_tau = float(np.maximum(piecewise_linear_extrap(np.array([tau]), np.array([30.0 / CALENDAR_DAYS_PER_YEAR, 90.0 / CALENDAR_DAYS_PER_YEAR]), np.array([state["IV30"], state["IV90"]]))[0], 1e-8))
    sigma = float(alpha_tau(np.array([tau]))[0] * iv_tau + (1.0 - alpha_tau(np.array([tau]))[0]) * state["RV20"])
    return {"date": date, "S": state["S"], "K": K, "k": k, "tau": tau, "r_tau": r, "sigma_teacher": sigma}


def predict_price_delta_batch(
    fitted: FittedSurrogate,
    df: pd.DataFrame,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    x_standardized: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict only price and Delta, avoiding unnecessary second derivatives."""
    model = fitted.model.to(device=device, dtype=dtype)
    model.eval()
    x_mean, x_scale = scaler_tensors(fitted.x_scaler, device, dtype)
    y_mean, y_scale = scaler_tensors(fitted.y_scaler, device, dtype)
    x_all = (
        fitted.x_scaler.transform(df[FEATURE_COLS].values.astype(float))
        if x_standardized is None
        else np.asarray(x_standardized, dtype=float)
    )
    prices: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    k_idx = FEATURE_COLS.index("k")
    for start in range(0, len(df), batch_size):
        stop = min(len(df), start + batch_size)
        x_base = torch.as_tensor(x_all[start:stop], device=device, dtype=dtype)
        if fitted.kind == "A":
            x = x_base.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                output = model(x)
                c, _, _ = price_potential_from_output(
                    fitted.kind, output, x, x_mean, x_scale, y_mean, y_scale
                )
                grad = torch.autograd.grad(c.sum(), x, create_graph=False, retain_graph=False)[0]
                c_k = grad[:, k_idx] / x_scale[k_idx]
                delta = c - c_k
        else:
            with torch.no_grad():
                output = model(x_base)
                c, direct_raw, _ = price_potential_from_output(
                    fitted.kind, output, x_base, x_mean, x_scale, y_mean, y_scale
                )
                delta = direct_raw[:, 1]
        S_np = df["S"].iloc[start:stop].to_numpy(dtype=float, copy=True)
        S = torch.as_tensor(S_np, device=device, dtype=dtype)
        prices.append((S * c).detach().cpu().numpy())
        deltas.append(delta.detach().cpu().numpy())
    return np.concatenate(prices), np.concatenate(deltas)


def build_hedging_state_cache(
    test_df: pd.DataFrame,
    daily: pd.DataFrame,
    max_paths: int,
    progress_every_paths: int = 5000,
) -> HedgingStateCache:
    """Construct all path-step states once, shared by every model and seed."""
    daily = daily.sort_values("Date").reset_index(drop=True)
    position = {pd.Timestamp(d): i for i, d in enumerate(daily["Date"])}
    lookup = daily_lookup(daily)
    candidates = test_df[test_df["high_curvature_region"] == 1].copy()
    keep: List[bool] = []
    for _, row in candidates.iterrows():
        p0 = position.get(pd.Timestamp(row["date"]))
        if p0 is None or p0 + 5 >= len(daily):
            keep.append(False)
            continue
        end_date = pd.Timestamp(daily.iloc[p0 + 5]["Date"])
        keep.append(remaining_tau(pd.Timestamp(row["date"]), end_date, float(row["tau"])) > 0.0)
    candidates = candidates.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)
    if max_paths > 0 and len(candidates) > max_paths:
        candidates = candidates.sample(max_paths, random_state=123).sort_values(["date", "tau", "k"]).reset_index(drop=True)

    n_paths = len(candidates)
    if n_paths == 0:
        return HedgingStateCache(
            states=pd.DataFrame(columns=["date", "S", "K", "k", "tau", "r_tau", "sigma_teacher"]),
            n_paths=0,
            spot=np.empty((0, 6)),
            rate=np.empty((0, 6)),
            dt=np.empty((0, 5)),
            teacher_final=np.empty(0),
        )

    rows: List[Dict[str, float]] = []
    spot = np.empty((n_paths, 6), dtype=float)
    rate = np.empty((n_paths, 6), dtype=float)
    dt = np.empty((n_paths, 5), dtype=float)
    hedge_grid_start = time.perf_counter()
    for path_id, contract in candidates.iterrows():
        if progress_every_paths > 0 and (
            path_id == 0 or (path_id + 1) % progress_every_paths == 0 or path_id + 1 == n_paths
        ):
            done = path_id + 1
            elapsed = time.perf_counter() - hedge_grid_start
            eta = elapsed * (n_paths - done) / max(done, 1)
            print(
                f"[hedging-grid] paths={done}/{n_paths} "
                f"progress={100.0 * done / n_paths:5.1f}% "
                f"elapsed={human_duration(elapsed)} eta~={human_duration(eta)}",
                flush=True,
            )
        start_date = pd.Timestamp(contract["date"])
        p0 = position[start_date]
        K = float(contract["K"])
        tau0 = float(contract["tau"])
        previous_date = start_date
        for step in range(6):
            current_date = pd.Timestamp(daily.iloc[p0 + step]["Date"])
            tau_now = remaining_tau(start_date, current_date, tau0)
            state_now = contract_state(current_date, K, tau_now, lookup[current_date])
            rows.append(state_now)
            spot[path_id, step] = state_now["S"]
            rate[path_id, step] = state_now["r_tau"]
            if step > 0:
                dt[path_id, step - 1] = max(
                    (current_date - previous_date).days / CALENDAR_DAYS_PER_YEAR,
                    0.0,
                )
            previous_date = current_date

    states = pd.DataFrame(rows)
    final = states.iloc[5::6].reset_index(drop=True)
    teacher_final = bsm_call_engine(
        final["S"].values,
        final["K"].values,
        final["r_tau"].values,
        final["tau"].values,
        final["sigma_teacher"].values,
    )["price"]
    return HedgingStateCache(
        states=states,
        n_paths=n_paths,
        spot=spot,
        rate=rate,
        dt=dt,
        teacher_final=np.asarray(teacher_final, dtype=float),
    )


def hedging_metrics_from_cache(
    cache: HedgingStateCache,
    models: Dict[str, FittedSurrogate],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> pd.DataFrame:
    """Vectorized five-day hedging over all contracts for each model."""
    if cache.n_paths == 0:
        return pd.DataFrame([
            {"model": name, "mean_pnl": np.nan, "std_pnl": np.nan,
             "abs_pnl_q95": np.nan, "es95_loss": np.nan, "n_contracts": 0}
            for name in models
        ])

    rows: List[Dict[str, Any]] = []
    hedge_start = time.perf_counter()
    n_models = len(models)
    for model_index, (name, fitted) in enumerate(models.items(), start=1):
        print(
            f"[hedging] model={model_index}/{n_models} name={name} started",
            flush=True,
        )
        price_flat, delta_flat = predict_price_delta_batch(
            fitted, cache.states, device, dtype, batch_size=batch_size
        )
        price = price_flat.reshape(cache.n_paths, 6)
        delta = delta_flat.reshape(cache.n_paths, 6)
        cash = price[:, 0] - delta[:, 0] * cache.spot[:, 0]
        delta_prev = delta[:, 0]
        portfolio = price[:, 0].copy()
        for step in range(1, 6):
            cash = cash * np.exp(cache.rate[:, step] * cache.dt[:, step - 1])
            portfolio = delta_prev * cache.spot[:, step] + cash
            cash = portfolio - delta[:, step] * cache.spot[:, step]
            delta_prev = delta[:, step]
        pnl = portfolio - cache.teacher_final
        rows.append({
            "model": name,
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl, ddof=0)),
            "abs_pnl_q95": float(np.quantile(np.abs(pnl), 0.95)),
            "es95_loss": expected_shortfall_loss(pnl),
            "n_contracts": int(len(pnl)),
        })
        elapsed = time.perf_counter() - hedge_start
        eta = elapsed * (n_models - model_index) / max(model_index, 1)
        print(
            f"[hedging] model={model_index}/{n_models} name={name} completed "
            f"elapsed={human_duration(elapsed)} eta~={human_duration(eta)}",
            flush=True,
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paper-facing exports
# ---------------------------------------------------------------------------

def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def format_number(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.3e}"
    return f"{x:.6f}"


def generate_paper_results(
    metrics: pd.DataFrame,
    consistency: pd.DataFrame,
    hedging: pd.DataFrame,
    paired: pd.DataFrame,
    seeds: Sequence[int],
    table_dir: Path,
) -> None:
    mean_metrics = metrics.groupby(["model", "region", "quantity"], as_index=False)["rmse"].mean()
    model_order = MODEL_NAMES
    quantities = ["price", "delta", "gamma", "vega", "vanna", "volga"]

    lines: List[str] = []
    row_end = chr(92) * 2
    lines.append("% Auto-generated by the structure-consistent empirical script.")
    for region, caption, label in [
        ("global", "Mean test-set RMSE across matched seeds", "tab:seds-global"),
        ("high_curvature", "Mean high-curvature-slice RMSE across matched seeds", "tab:seds-local"),
    ]:
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\small",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Model & Price & Delta & Gamma & Vega & Vanna & Volga \\\\",
            "\\midrule",
        ])
        for model in model_order:
            row = [model.replace("Model_", "Model ")]
            for quantity in quantities:
                frag = mean_metrics[(mean_metrics["model"] == model) & (mean_metrics["region"] == region) & (mean_metrics["quantity"] == quantity)]
                row.append(format_number(float(frag["rmse"].iloc[0])) if not frag.empty else "--")
            lines.append(" & ".join(row) + " " + row_end)
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if not consistency.empty:
        cmean = consistency.groupby(["model", "consistency_quantity"], as_index=False)["gap_rmse"].mean()
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Mean internal-consistency diagnostics across matched seeds}",
            "\\label{tab:seds-consistency}",
            "\\small",
            "\\begin{tabular}{llr}",
            "\\toprule",
            "Model & Diagnostic & RMSE \\\\",
            "\\midrule",
        ])
        for _, row in cmean.iterrows():
            lines.append(f"{row['model'].replace('Model_', 'Model ')} & {latex_escape(row['consistency_quantity'])} & {format_number(float(row['gap_rmse']))} " + row_end)
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if not hedging.empty:
        hmean = hedging.groupby("model", as_index=False).agg({"mean_pnl": "mean", "std_pnl": "mean", "abs_pnl_q95": "mean", "es95_loss": "mean", "n_contracts": "mean"})
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Mean five-day hedging diagnostics across matched seeds}",
            "\\label{tab:seds-hedging}",
            "\\small",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Model & Mean P\\&L & P\\&L SD & $|\\mathrm{P\\&L}|_{0.95}$ & ES-style loss & Contracts \\\\",
            "\\midrule",
        ])
        for model in model_order:
            frag = hmean[hmean["model"] == model]
            if frag.empty:
                continue
            row = frag.iloc[0]
            lines.append(
                f"{model.replace('Model_', 'Model ')} & {format_number(float(row['mean_pnl']))} & "
                f"{format_number(float(row['std_pnl']))} & {format_number(float(row['abs_pnl_q95']))} & "
                f"{format_number(float(row['es95_loss']))} & {int(round(float(row['n_contracts'])))} " + row_end)
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if not paired.empty:
        lines.extend([
            "\\scriptsize",
            "\\begin{longtable}{lllrlr}",
            "\\caption{Paired RMSE improvements across matched seeds; positive values favor the treatment model}\\label{tab:seds-paired}" + row_end,
            "\\toprule",
            "Slice & Quantity & Comparison & Mean & 95\\% CI & $p$-value " + row_end,
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "Slice & Quantity & Comparison & Mean & 95\\% CI & $p$-value " + row_end,
            "\\midrule",
            "\\endhead",
        ])
        order = {"global": 0, "high_curvature": 1}
        paired_sorted = paired.assign(_order=paired["region"].map(order)).sort_values(["_order", "quantity", "baseline", "treatment"])
        for _, row in paired_sorted.iterrows():
            slice_label = "Global" if row["region"] == "global" else "HC"
            comparison = row["treatment"].replace("Model_", "") + "--" + row["baseline"].replace("Model_", "")
            ci_text = "[" + format_number(float(row["ci_lower"])) + ", " + format_number(float(row["ci_upper"])) + "]"
            lines.append(
                f"{slice_label} & {latex_escape(row['quantity'])} & {comparison} & "
                f"{format_number(float(row['mean_improvement']))} & {ci_text} & "
                f"{format_number(float(row['p_value_two_sided']))} " + row_end
            )
        lines.extend(["\\bottomrule", "\\end{longtable}", "\\normalsize", ""])

    (table_dir / "paper_results_tables.tex").write_text("\n".join(lines), encoding="utf-8")

    global_price = mean_metrics[(mean_metrics["region"] == "global") & (mean_metrics["quantity"] == "price")]
    global_delta = mean_metrics[(mean_metrics["region"] == "global") & (mean_metrics["quantity"] == "delta")]
    high_delta = mean_metrics[(mean_metrics["region"] == "high_curvature") & (mean_metrics["quantity"] == "delta")]
    best_price = global_price.sort_values("rmse").iloc[0]["model"] if not global_price.empty else "Model_D"
    best_delta = global_delta.sort_values("rmse").iloc[0]["model"] if not global_delta.empty else "Model_D"
    best_high_delta = high_delta.sort_values("rmse").iloc[0]["model"] if not high_delta.empty else "Model_D"
    sentence = (
        f"Across {len(seeds)} matched seeds, {best_price.replace('_', ' ')} has the lowest mean global price RMSE, "
        f"{best_delta.replace('_', ' ')} has the lowest mean global Delta RMSE, and "
        f"{best_high_delta.replace('_', ' ')} has the lowest mean Delta RMSE on the high-curvature diagnostic slice. "
        "Model rankings remain metric dependent and are interpreted jointly with consistency and hedging diagnostics."
    )
    macros = [
        "% Auto-generated empirical macros.",
        "\\newcommand{\\EmpiricalAbstractSentence}{" + sentence + "}",
        "\\newcommand{\\EmpiricalSeedCount}{" + str(len(seeds)) + "}",
    ]
    (table_dir / "paper_results_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def numerical_diagnostics_table(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    np_dtype = resolve_numpy_dtype(args.dtype)
    training_dtype = args.training_dtype or args.dtype
    return pd.DataFrame([
        {"item": "teacher_dtype", "value": args.dtype, "comment": "Analytic teacher and derivative audits."},
        {"item": "training_dtype", "value": training_dtype, "comment": "Torch parameters and tensors."},
        {"item": "machine_epsilon", "value": f"{np.finfo(np_dtype).eps:.18e}", "comment": "Floating-point roundoff scale."},
        {"item": "device", "value": str(device), "comment": "Training device."},
        {"item": "activation", "value": "SiLU", "comment": "Smooth activation supports nontrivial second automatic derivatives."},
        {"item": "state_features", "value": ",".join(FEATURE_COLS), "comment": "Sufficient BSM state coordinates used by every model."},
        {"item": "normalized_targets", "value": ",".join(TARGET_LABELS), "comment": "Scale-controlled price and sensitivity coordinates."},
        {"item": "first_fd_scale", "value": "eps^(1/5)*max(1,|x|)", "comment": "Fourth-order first-derivative audit."},
        {"item": "second_fd_scale", "value": "eps^(1/6)*max(1,|x|)", "comment": "Fourth-order second-derivative audit."},
        {"item": "mixed_fd_scale", "value": "eps^(1/4)*max(1,|x|)", "comment": "Centered mixed-derivative audit."},
        {"item": "gradient_clip", "value": str(args.gradient_clip), "comment": "Global norm clipping during optimization."},
        {"item": "validation_rule", "value": "shared standardized six-target RMSE", "comment": "Identical selection score for Models A-D."},
        {"item": "matched_seeds", "value": args.seeds, "comment": "All models use the same seed list."},
        {"item": "local_slice_role", "value": "evaluation_only", "comment": "Short-maturity near-the-money observations never alter training weights or sampling."},
        {"item": "processed_cache", "value": str(args.reuse_processed_cache), "comment": "Reusable point-in-time panel and teacher audits."},
        {"item": "resume", "value": str(args.resume), "comment": "Completed models, epoch checkpoints, evaluations, and hedging outputs are reusable."},
        {"item": "validation_interval", "value": str(args.validation_interval), "comment": "Full validation frequency in epochs."},
        {"item": "max_epochs", "value": str(args.max_epochs), "comment": "Common epoch cap for Models B-D; default 320."},
        {"item": "model_a_max_epochs", "value": str(args.model_a_max_epochs or args.max_epochs), "comment": "Optional extended epoch cap for Model A."},
        {"item": "progress_reporting", "value": "weighted model/overall ETA", "comment": "Epoch, model, seed, hedging-grid, and pipeline-stage progress."},
        {"item": "vectorized_hedging", "value": "True", "comment": "All path-step states are batch-evaluated per model."},
    ])


def save_fitted_model(fitted: FittedSurrogate, path: Path) -> None:
    payload = {
        "name": fitted.name,
        "kind": fitted.kind,
        "state_dict": {k: v.detach().cpu() for k, v in fitted.model.state_dict().items()},
        "x_mean": fitted.x_scaler.mean_,
        "x_scale": fitted.x_scaler.scale_,
        "y_mean": fitted.y_scaler.mean_,
        "y_scale": fitted.y_scaler.scale_,
        "config": asdict(fitted.config),
        "seed": fitted.seed,
        "best_epoch": fitted.best_epoch,
        "best_validation_score": fitted.best_validation_score,
        "dtype_name": fitted.dtype_name,
        "feature_cols": fitted.feature_cols,
        "target_cols": fitted.target_cols,
    }
    torch.save(payload, path)


def load_fitted_model(
    path: Path,
    expected_name: str,
    expected_kind: str,
    expected_seed: int,
    expected_config: TrainConfig,
    expected_dtype_name: str,
    device: torch.device,
) -> Optional[FittedSurrogate]:
    """Load a completed compatible model for resume-by-model caching."""
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checks = [
            payload.get("name") == expected_name,
            payload.get("kind") == expected_kind,
            int(payload.get("seed", -1)) == int(expected_seed),
            payload.get("dtype_name") == expected_dtype_name,
            payload.get("feature_cols") == list(FEATURE_COLS),
            payload.get("target_cols") == list(TARGET_COLS),
            payload.get("config") == asdict(expected_config),
        ]
        if not all(checks):
            return None
        model = SmoothSurrogate(len(FEATURE_COLS), expected_config.hidden_dim, expected_config.depth, expected_kind)
        model.load_state_dict(payload["state_dict"])
        dtype = resolve_torch_dtype(expected_dtype_name)
        model.to(device=device, dtype=dtype)

        x_scaler = StandardScaler()
        x_scaler.mean_ = np.asarray(payload["x_mean"], dtype=float)
        x_scaler.scale_ = np.asarray(payload["x_scale"], dtype=float)
        x_scaler.var_ = x_scaler.scale_ ** 2
        x_scaler.n_features_in_ = len(x_scaler.mean_)
        y_scaler = StandardScaler()
        y_scaler.mean_ = np.asarray(payload["y_mean"], dtype=float)
        y_scaler.scale_ = np.asarray(payload["y_scale"], dtype=float)
        y_scaler.var_ = y_scaler.scale_ ** 2
        y_scaler.n_features_in_ = len(y_scaler.mean_)
        return FittedSurrogate(
            name=expected_name,
            kind=expected_kind,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            config=expected_config,
            seed=expected_seed,
            best_epoch=int(payload["best_epoch"]),
            best_validation_score=float(payload["best_validation_score"]),
            dtype_name=expected_dtype_name,
            feature_cols=list(FEATURE_COLS),
            target_cols=list(TARGET_COLS),
        )
    except Exception as exc:
        warnings.warn(f"Could not load completed model cache {path}: {exc}", RuntimeWarning)
        return None


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content fingerprint for a chronological option split."""
    cols = ["date", "S", "K", "k", "tau", "r_tau", "sigma_teacher"]
    hashed = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_evaluation_cache(
    cache_dir: Path,
    model_path: Path,
    test_signature: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    meta_path = cache_dir / "meta.json"
    metrics_path = cache_dir / "metrics.pkl"
    consistency_path = cache_dir / "consistency.pkl"
    if not (meta_path.exists() and metrics_path.exists() and consistency_path.exists() and model_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_file") != file_fingerprint(model_path):
            return None
        if meta.get("test_signature") != test_signature:
            return None
        return pd.read_pickle(metrics_path), pd.read_pickle(consistency_path)
    except Exception:
        return None


def save_evaluation_cache(
    cache_dir: Path,
    model_path: Path,
    test_signature: str,
    metrics: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    ensure_dir(cache_dir)
    metrics.to_pickle(cache_dir / "metrics.pkl")
    consistency.to_pickle(cache_dir / "consistency.pkl")
    meta = {"model_file": file_fingerprint(model_path), "test_signature": test_signature}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Command line and pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Greek- and structure-consistent SEDS empirical study.")
    p.add_argument("--output-dir", default="SEDS_structure_outputs")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--mock-data", action="store_true")
    p.add_argument("--n-mock-days", type=int, default=1200)
    p.add_argument("--analysis-start", default=None)
    p.add_argument("--analysis-end", default=None)
    p.add_argument("--max-source-age-days", type=int, default=10)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--training-dtype",
        choices=["float64", "float32"],
        default=None,
        help="Optional training dtype; teacher labels and audits continue to use --dtype.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seeds", default="1234,2234,3234,4234,5234")
    p.add_argument("--cpu-threads", type=int, default=0, help="0 uses all visible CPU cores.")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preload-training-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-preload-gb", type=float, default=8.0)
    p.add_argument("--validation-interval", type=int, default=1)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse-processed-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-dir", default=None, help="Defaults to <output-dir>/cache.")

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=320)
    p.add_argument(
        "--model-a-max-epochs",
        type=int,
        default=0,
        help="Optional Model A epoch cap; 0 uses --max-epochs. This allows an extended A-only convergence run while keeping Models B-D cache-compatible.",
    )
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--consistency-fraction", type=float, default=0.25)

    p.add_argument("--lambda-delta", type=float, default=1.0)
    p.add_argument("--lambda-gamma", type=float, default=2.0)
    p.add_argument("--lambda-vega", type=float, default=1.0)
    p.add_argument("--lambda-vanna", type=float, default=1.0)
    p.add_argument("--lambda-volga", type=float, default=1.0)
    p.add_argument("--lambda-potential", type=float, default=1.0)
    p.add_argument("--lambda-decomposition", type=float, default=1.0)
    p.add_argument("--lambda-weights", type=float, default=0.5)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--max-hedge-paths", type=int, default=0)
    p.add_argument("--progress-path-every", type=int, default=5000, help="Hedging-grid progress interval; 0 disables path progress.")
    p.add_argument("--skip-hedging", action="store_true")
    p.add_argument("--teacher-audit-samples", type=int, default=32)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--save-full-panel", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    device = resolve_device(args.device)
    run_progress = RunProgress(n_seeds=len(seeds), start_time=run_start)
    training_dtype_name = args.training_dtype or args.dtype
    torch_dtype = resolve_torch_dtype(training_dtype_name)
    numpy_dtype = resolve_numpy_dtype(args.dtype)
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1).")
    if args.cpu_threads < 0:
        raise ValueError("cpu_threads must be nonnegative.")
    cpu_threads = args.cpu_threads or (os.cpu_count() or 1)
    torch.set_num_threads(max(1, cpu_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass
    if args.log_every < 0:
        raise ValueError("log_every must be nonnegative.")
    if args.progress_path_every < 0:
        raise ValueError("progress_path_every must be nonnegative.")

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"[run] SEDS started device={device} hardware={gpu_name} "
        f"teacher_dtype={args.dtype} training_dtype={args.training_dtype or args.dtype}",
        flush=True,
    )
    print(
        f"[run] seeds={len(seeds)} models={len(seeds) * len(MODEL_NAMES)} "
        f"max_epochs={args.max_epochs} model_a_max_epochs={args.model_a_max_epochs or args.max_epochs} patience={args.patience} "
        f"validation_interval={args.validation_interval} batch_size={args.batch_size}",
        flush=True,
    )

    if device.type == "cuda" and args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        warnings.warn(
            "Strict deterministic CUDA execution is requested but CUBLAS_WORKSPACE_CONFIG is not set. "
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.",
            RuntimeWarning,
        )

    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        gradient_clip=args.gradient_clip,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        consistency_fraction=args.consistency_fraction,
        lambda_delta=args.lambda_delta,
        lambda_gamma=args.lambda_gamma,
        lambda_vega=args.lambda_vega,
        lambda_vanna=args.lambda_vanna,
        lambda_volga=args.lambda_volga,
        lambda_potential=args.lambda_potential,
        lambda_decomposition=args.lambda_decomposition,
        lambda_weights=args.lambda_weights,
    )

    out_dir = Path(args.output_dir)
    data_out = out_dir / "data"
    table_out = out_dir / "tables"
    model_out = out_dir / "models"
    history_out = out_dir / "history"
    checkpoint_out = out_dir / "checkpoints"
    cache_out = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    evaluation_cache_out = cache_out / "evaluation"
    for path in [out_dir, data_out, table_out, model_out, history_out, checkpoint_out, cache_out, evaluation_cache_out]:
        ensure_dir(path)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    signature = processed_cache_signature(args, data_dir, seeds[0])
    run_progress.stage("data/cache lookup")
    cached = load_processed_cache(cache_out / "processed", signature) if args.reuse_processed_cache else None
    if cached is not None:
        daily, panel, source_audit, validation, audit = cached
        print("[cache] loaded processed daily state, teacher panel, and audits", flush=True)
    else:
        stage_start = time.perf_counter()
        raw, source_audit = load_daily_market_data(
            use_mock=args.mock_data,
            n_mock_days=args.n_mock_days,
            seed=seeds[0],
            data_dir=data_dir,
            analysis_start=args.analysis_start,
            analysis_end=args.analysis_end,
            max_source_age_days=args.max_source_age_days,
        )
        daily = prepare_daily_state(raw)
        panel = assign_splits(build_teacher_panel(daily, dtype=numpy_dtype), use_mock=args.mock_data)
        validation = data_validation_table(daily, panel, source_audit)
        if not validation["passed"].all():
            failed = validation.loc[~validation["passed"], ["check", "detail"]].to_dict("records")
            raise ValueError(f"Data/teacher validation failed: {failed}")
        audit = teacher_derivative_audit(panel, args.teacher_audit_samples, seeds[0], dtype=numpy_dtype)
        if args.reuse_processed_cache:
            save_processed_cache(
                cache_out / "processed", signature, daily, panel, source_audit, validation, audit
            )
        print(f"[data] prepared panel in {(time.perf_counter() - stage_start) / 60.0:.1f} min", flush=True)

    run_progress.stage("writing data and validation outputs")
    daily.to_csv(data_out / "daily_market_state.csv", index=False)
    source_audit.to_csv(table_out / "table_source_coverage.csv", index=False)
    validation.to_csv(table_out / "table_data_validation.csv", index=False)
    audit.to_csv(table_out / "table_teacher_derivative_audit.csv", index=False)
    panel_csv_path = data_out / "teacher_panel_full.csv"
    if args.save_full_panel and not (args.resume and panel_csv_path.exists()):
        panel.to_csv(panel_csv_path, index=False)

    sample_counts = panel.groupby("split").agg(
        n_samples=("date", "size"),
        n_dates=("date", "nunique"),
        n_high_curvature=("high_curvature_region", "sum"),
        start_date=("date", "min"),
        end_date=("date", "max"),
    ).reset_index()
    sample_counts.to_csv(table_out / "table_sample_counts.csv", index=False)
    numerical_diagnostics_table(args, device).to_csv(table_out / "table_numerical_diagnostics.csv", index=False)

    all_metrics: List[pd.DataFrame] = []
    all_consistency: List[pd.DataFrame] = []
    all_hedging: List[pd.DataFrame] = []
    selection_rows: List[Dict[str, Any]] = []
    primary_prediction_files: List[str] = []
    shared_prepared: Optional[PreparedSeedData] = None
    shared_hedging_cache: Optional[HedgingStateCache] = None
    can_share_splits = args.max_train_samples <= 0 and args.max_val_samples <= 0 and args.max_test_samples <= 0

    run_progress.stage("matched-seed training and evaluation")
    for seed_index, seed in enumerate(seeds):
        print(f"[seed={seed}] seed={seed_index + 1}/{len(seeds)} preparing matched data", flush=True)
        if can_share_splits and shared_prepared is not None:
            prepared = shared_prepared
        else:
            prepared = prepare_seed_data(panel, seed, args, device, torch_dtype)
            if can_share_splits:
                shared_prepared = prepared
        test_signature = frame_fingerprint(prepared.test_df)
        fitted_models: Dict[str, FittedSurrogate] = {}

        for model_index, (model_name, kind) in enumerate(zip(MODEL_NAMES, ["A", "B", "C", "D"]), start=1):
            model_config = (
                replace(config, max_epochs=args.model_a_max_epochs)
                if model_name == "Model_A" and args.model_a_max_epochs > 0
                else config
            )
            print(
                f"[model] seed={seed_index + 1}/{len(seeds)} model={model_index}/4 "
                f"name={model_name} kind={kind} max_epochs={model_config.max_epochs} started",
                flush=True,
            )
            model_path = model_out / f"{model_name}_seed{seed}.pt"
            history_path = history_out / f"history_{model_name}_seed{seed}.csv"
            checkpoint_path = checkpoint_out / f"checkpoint_{model_name}_seed{seed}.pt"
            fitted: Optional[FittedSurrogate] = None
            history = pd.DataFrame()
            loaded_from_cache = False
            if args.resume:
                fitted = load_fitted_model(
                    model_path,
                    model_name,
                    kind,
                    seed,
                    model_config,
                    training_dtype_name,
                    device,
                )
                if fitted is not None:
                    scaler_match = (
                        np.allclose(fitted.x_scaler.mean_, prepared.x_scaler.mean_, rtol=0.0, atol=1e-12)
                        and np.allclose(fitted.x_scaler.scale_, prepared.x_scaler.scale_, rtol=0.0, atol=1e-12)
                        and np.allclose(fitted.y_scaler.mean_, prepared.y_scaler.mean_, rtol=0.0, atol=1e-12)
                        and np.allclose(fitted.y_scaler.scale_, prepared.y_scaler.scale_, rtol=0.0, atol=1e-12)
                    )
                    if not scaler_match:
                        warnings.warn(f"Ignoring stale completed model {model_path}: scaler mismatch.", RuntimeWarning)
                        fitted = None
                if fitted is not None and history_path.exists():
                    history = pd.read_csv(history_path)
                    loaded_from_cache = True
                    print(f"[cache] loaded completed {model_name} seed={seed}", flush=True)

            if fitted is None:
                fitted, history = fit_one_model(
                    model_name,
                    kind,
                    prepared,
                    seed,
                    model_config,
                    training_dtype_name,
                    device,
                    args.deterministic,
                    validation_interval=args.validation_interval,
                    log_every=args.log_every,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=args.checkpoint_every,
                    resume_checkpoint=args.resume,
                    run_progress=run_progress,
                )
                history.to_csv(history_path, index=False)
                save_fitted_model(fitted, model_path)

            fitted_models[model_name] = fitted
            final_epoch = int(history["epoch"].max()) if not history.empty and "epoch" in history else int(fitted.best_epoch)
            hit_max_epochs = bool(final_epoch >= model_config.max_epochs)
            selection_rows.append({
                "model": model_name,
                "seed": seed,
                "final_epoch": final_epoch,
                "best_epoch": fitted.best_epoch,
                "best_validation_score": fitted.best_validation_score,
                "hit_max_epochs": hit_max_epochs,
                "stopping_reason": "max_epochs" if hit_max_epochs else "early_stopping_or_cached",
                "best_epoch_within_last_10": bool(fitted.best_epoch >= max(1, final_epoch - 9)),
            })

            eval_cache_dir = evaluation_cache_out / f"{model_name}_seed{seed}"
            cached_eval = load_evaluation_cache(eval_cache_dir, model_path, test_signature) if args.resume else None
            primary_path = data_out / f"predictions_{model_name}_test_primary_seed.csv"
            need_primary_prediction = seed_index == 0 and not primary_path.exists()
            if cached_eval is not None and not need_primary_prediction:
                metric_df, cdf = cached_eval
                print(f"[cache] loaded evaluation {model_name} seed={seed}", flush=True)
            else:
                eval_start = time.perf_counter()
                pred = prediction_frame(
                    fitted,
                    prepared.test_df,
                    device,
                    torch_dtype,
                    batch_size=max(64, args.evaluation_batch_size),
                    x_standardized=prepared.test_x_std,
                )
                if seed_index == 0:
                    pred.to_csv(primary_path, index=False)
                    primary_prediction_files.append(str(primary_path))
                metric_rows = (
                    metric_rows_for_frame(pred, model_name, "global")
                    + metric_rows_for_frame(pred, model_name, "high_curvature")
                )
                metric_df = pd.DataFrame(metric_rows)
                metric_df.insert(0, "seed", seed)
                cdf = pd.DataFrame(consistency_metrics(pred, model_name))
                if not cdf.empty:
                    cdf.insert(0, "seed", seed)
                save_evaluation_cache(eval_cache_dir, model_path, test_signature, metric_df, cdf)
                print(
                    f"[evaluation] {model_name} seed={seed} completed in "
                    f"{(time.perf_counter() - eval_start) / 60.0:.1f} min",
                    flush=True,
                )
            if seed_index == 0 and primary_path.exists() and str(primary_path) not in primary_prediction_files:
                primary_prediction_files.append(str(primary_path))
            all_metrics.append(metric_df)
            if not cdf.empty:
                all_consistency.append(cdf)
            run_progress.mark_model_complete(
                kind, f"{model_name} seed={seed}", cached=loaded_from_cache
            )

        if not args.skip_hedging:
            hedge_cache_path = cache_out / f"hedging_states_{test_signature}_max{args.max_hedge_paths}.pkl"
            if can_share_splits and shared_hedging_cache is not None:
                hedge_state_cache = shared_hedging_cache
            elif args.resume and hedge_cache_path.exists():
                hedge_state_cache = pd.read_pickle(hedge_cache_path)
                print(f"[cache] loaded hedging state grid for seed={seed}", flush=True)
            else:
                hedge_state_cache = build_hedging_state_cache(
                    prepared.test_df, daily, args.max_hedge_paths,
                    progress_every_paths=args.progress_path_every,
                )
                pd.to_pickle(hedge_state_cache, hedge_cache_path)
                print(
                    f"[hedging] cached {hedge_state_cache.n_paths} paths x 6 states",
                    flush=True,
                )
            if can_share_splits:
                shared_hedging_cache = hedge_state_cache

            hedge_result_path = cache_out / f"hedging_result_seed{seed}.pkl"
            hedge_meta_path = cache_out / f"hedging_result_seed{seed}.json"
            model_signature = {
                name: file_fingerprint(model_out / f"{name}_seed{seed}.pt") for name in MODEL_NAMES
            }
            hedge_meta = {
                "test_signature": test_signature,
                "models": model_signature,
                "max_hedge_paths": args.max_hedge_paths,
            }
            hedge: Optional[pd.DataFrame] = None
            if args.resume and hedge_result_path.exists() and hedge_meta_path.exists():
                try:
                    if json.loads(hedge_meta_path.read_text(encoding="utf-8")) == hedge_meta:
                        hedge = pd.read_pickle(hedge_result_path)
                        print(f"[cache] loaded hedging result seed={seed}", flush=True)
                except Exception:
                    hedge = None
            if hedge is None:
                hedge_start = time.perf_counter()
                hedge = hedging_metrics_from_cache(
                    hedge_state_cache,
                    fitted_models,
                    device,
                    torch_dtype,
                    batch_size=max(64, args.evaluation_batch_size),
                )
                hedge.to_pickle(hedge_result_path)
                hedge_meta_path.write_text(json.dumps(hedge_meta, indent=2), encoding="utf-8")
                print(
                    f"[hedging] seed={seed} completed in {(time.perf_counter() - hedge_start) / 60.0:.1f} min",
                    flush=True,
                )
            hedge.insert(0, "seed", seed)
            all_hedging.append(hedge)

    run_progress.stage("aggregation and paper-facing exports")
    metrics = pd.concat(all_metrics, ignore_index=True)
    consistency = pd.concat(all_consistency, ignore_index=True) if all_consistency else pd.DataFrame()
    hedging = pd.concat(all_hedging, ignore_index=True) if all_hedging else pd.DataFrame()
    selection = pd.DataFrame(selection_rows)
    paired = paired_metric_summary(metrics, args.confidence_level)

    metrics.to_csv(table_out / "table_metrics_per_seed.csv", index=False)
    metrics.groupby(["model", "region", "quantity"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        sd_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        n_seeds=("seed", "nunique"),
    ).to_csv(table_out / "table_metrics_summary.csv", index=False)
    paired.to_csv(table_out / "table_paired_inference.csv", index=False)
    selection.to_csv(table_out / "table_model_selection.csv", index=False)
    selection.to_csv(table_out / "table_convergence_diagnostics.csv", index=False)
    n_hit_cap = int(selection["hit_max_epochs"].sum()) if "hit_max_epochs" in selection else 0
    print(
        f"[convergence] models_hitting_max_epochs={n_hit_cap}/{len(selection)}; "
        f"models_stopped_before_cap={len(selection) - n_hit_cap}/{len(selection)}",
        flush=True,
    )
    if not consistency.empty:
        consistency.to_csv(table_out / "table_consistency_per_seed.csv", index=False)
        consistency.groupby(["model", "consistency_quantity"], as_index=False).agg(
            mean_gap_rmse=("gap_rmse", "mean"),
            sd_gap_rmse=("gap_rmse", "std"),
            mean_gap_mae=("gap_mae", "mean"),
            n_seeds=("seed", "nunique"),
        ).to_csv(table_out / "table_consistency_summary.csv", index=False)
    if not hedging.empty:
        hedging.to_csv(table_out / "table_hedging_per_seed.csv", index=False)
        hedging.groupby("model", as_index=False).agg(
            mean_pnl=("mean_pnl", "mean"),
            mean_std_pnl=("std_pnl", "mean"),
            mean_abs_pnl_q95=("abs_pnl_q95", "mean"),
            mean_es95_loss=("es95_loss", "mean"),
            mean_n_contracts=("n_contracts", "mean"),
        ).to_csv(table_out / "table_hedging_summary.csv", index=False)

    generate_paper_results(metrics, consistency, hedging, paired, seeds, table_out)

    manifest = {
        "study": "Greek- and structure-consistent sensitivity-engine derivative surrogates",
        "implementation": "Extended-convergence cached/vectorized execution with progress reporting",
        "models": {
            "Model_A": "normalized-price potential; sensitivities from automatic derivatives",
            "Model_B": "direct price and Greek heads",
            "Model_C": "direct heads plus price-potential consistency",
            "Model_D": "potential consistency plus two-leg decomposition and supervised digital weights",
        },
        "training": {
            "common_models_B_to_D": asdict(config),
            "Model_A": asdict(replace(config, max_epochs=args.model_a_max_epochs or config.max_epochs)),
        },
        "seeds": seeds,
        "teacher_dtype": args.dtype,
        "training_dtype": training_dtype_name,
        "device": str(device),
        "data_dir_used": str(data_dir.resolve()),
        "expected_local_files": list(LOCAL_SOURCE_FILENAMES.values()),
        "high_curvature_slice": {
            "tau_max": HIGH_CURVATURE_TAU_MAX,
            "abs_k_max": HIGH_CURVATURE_K_MAX,
            "role": "evaluation_only",
        },
        "execution_optimizations": {
            "processed_cache": args.reuse_processed_cache,
            "completed_model_resume": args.resume,
            "epoch_checkpoint_resume": args.resume,
            "shared_standardized_tensors": True,
            "preload_training_data": args.preload_training_data,
            "validation_interval": args.validation_interval,
            "vectorized_hedging": True,
            "evaluation_cache": args.resume,
            "weighted_progress_eta": True,
            "hedging_grid_progress_every_paths": args.progress_path_every,
        },
        "primary_prediction_files": primary_prediction_files,
        "paper_includes": [
            str(table_out / "paper_results_macros.tex"),
            str(table_out / "paper_results_tables.tex"),
        ],
        "args": vars(args),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    run_progress.finish()
    print(f"Finished. Outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
