from __future__ import annotations

"""
Sensitivity-Engine Derivative Surrogates

Empirical implementation for the free-data study described in
"Sensitivity-Engine Derivative Surrogates: Greek-Regularized and Boundary-Aware Supervision."

Local data mode
---------------
This version reads the required CSV files from the SAME folder as the main program
by default. The expected filenames are:
  - spx_stooq.csv
  - DGS3MO.csv
  - DGS2.csv
  - DGS10.csv
  - VIXCLS.csv
  - VXVCLS.csv

Original source links for those local files
-------------------------------------------
Stooq SPX historical CSV:
  https://stooq.com/q/d/?s=%5Espx
  https://stooq.com/q/d/l/?s=%5Espx&i=d
FRED CSV endpoints:
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXVCLS

What the script does
--------------------
1. Reads the local daily market data files listed above.
2. Builds the daily state variables in the SEDS empirical design.
3. Generates the synthetic teacher-label panel on the fixed strike/maturity grid.
4. Trains three surrogate models:
   - Model A: price-only MLP
   - Model B: Greek-regularized multi-output MLP
   - Model C: boundary-aware Greek-regularized multi-output MLP
5. Outputs all paper-facing data and tables as CSV files.

Notes
-----
- The teacher is a standalone Black-Scholes-Merton call engine consistent with the
  BSE analytic setting.
- Dividend yield q is fixed at 0.0 because the empirical subsection does not introduce
  a dividend-yield proxy.
- Model A prices only. Its delta and gamma are recovered by finite differences in k,
  then mapped back to S-derivatives using K = S exp(k).
- Models B and C are trained on [price, sqrt(lambda_delta)*delta, sqrt(lambda_gamma)*asinh(S^2 gamma)].
  This stabilizes the Gamma target while keeping the weighted-loss interpretation on the transformed head.
- Structured models are selected on the validation set with a composite score that gives
  explicit weight to local boundary-region price RMSE while retaining Delta and Gamma objectives.
- Model C additionally uses a softer, C-specific candidate search over boundary intensity,
  boundary width, and Gamma pressure so local price fit can improve without changing Model B.
- The script still includes --mock-data mode for dry runs, but the default workflow is
  local-file loading rather than live downloading.
"""

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as _normal_cdf
except Exception:  # pragma: no cover - fallback for minimal environments
    _normal_cdf = None
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Global constants used throughout the empirical design
# ----------------------------------------------------------------------------
# These objects encode the fixed grid and local region definition used in 
# SEDS.  The teacher panel is built on the Cartesian product of:
#   - 9 maturities (TAU_GRID_DAYS / TAU_GRID_YEARS)
#   - 15 log-moneyness points (K_GRID)
# The "boundary region" is the short-dated, near-the-money region that receives
# extra empirical attention in SEDS.
# ============================================================================

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
TAU_GRID_DAYS = np.array([3, 5, 7, 14, 21, 30, 45, 60, 90], dtype=float)
TAU_GRID_YEARS = TAU_GRID_DAYS / CALENDAR_DAYS_PER_YEAR
K_GRID = np.array([-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15], dtype=float)
BOUNDARY_TAU_MAX = 30.0 / CALENDAR_DAYS_PER_YEAR
BOUNDARY_K_MAX = 0.05

LOCAL_SOURCE_FILENAMES = {
    "spx": "spx_stooq.csv",
    "DGS3MO": "DGS3MO.csv",
    "DGS2": "DGS2.csv",
    "DGS10": "DGS10.csv",
    "VIXCLS": "VIXCLS.csv",
    "VXVCLS": "VXVCLS.csv",
}


def resolve_dtype(dtype_name: str):
    """Return the NumPy floating dtype requested from the command line.

    The empirical tables in the paper use double precision.  The explicit dtype
    switch is included for numerical-diagnostics and reproducibility checks.
    """
    mapping = {"float64": np.float64, "float32": np.float32}
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Use one of {sorted(mapping)}.")
    return mapping[dtype_name]


def as_float_array(x, dtype=np.float64) -> np.ndarray:
    """Convert input to a NumPy floating array with the selected dtype."""
    return np.asarray(x, dtype=dtype)


def machine_epsilon(dtype=np.float64) -> float:
    """Machine epsilon for the selected floating dtype."""
    return float(np.finfo(dtype).eps)


def default_data_dir() -> Path:
    """Return the folder that contains this script.

    In the default local-data workflow, the input CSV files are expected to sit
    next to the program itself, so this helper gives the natural fallback path.
    """
    return Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a possibly messy text column into numeric form.

    This is used for CSV inputs where values may contain commas or non-numeric
    placeholders.  Invalid entries become NaN and can be filtered later.
    """
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def norm_pdf(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Standard normal probability density function in vectorized NumPy form."""
    x = as_float_array(x, dtype)
    two = np.asarray(2.0, dtype=dtype)
    pi = np.asarray(np.pi, dtype=dtype)
    return np.exp(-np.asarray(0.5, dtype=dtype) * x * x, dtype=dtype) / np.sqrt(two * pi, dtype=dtype)


def norm_cdf(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Standard normal cumulative distribution function.

    A scipy.special.ndtr backend is used when available because it is a stable
    vectorized implementation in the tails.  The fallback keeps the script
    runnable in minimal environments.
    """
    x = as_float_array(x, dtype)
    if _normal_cdf is not None:
        return np.asarray(_normal_cdf(x), dtype=dtype)
    return np.asarray(0.5, dtype=dtype) * (
        np.asarray(1.0, dtype=dtype) + np.vectorize(math.erf)(x / np.sqrt(np.asarray(2.0, dtype=dtype)))
    )


def safe_exp(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Exponentiate after clipping to the representable range of the dtype."""
    x = as_float_array(x, dtype)
    finfo = np.finfo(dtype)
    return np.exp(np.clip(x, np.log(finfo.tiny), np.log(finfo.max)), dtype=dtype)


def fd_base_step(dtype=np.float64, requested_step: float = 1e-3) -> float:
    """Base finite-difference step for the price-only Model A Greeks.

    Fourth-order centered first derivatives have an ideal scale of eps^(1/5),
    while fourth-order centered second derivatives have an ideal scale closer to
    eps^(1/6).  Gamma is the more sensitive diagnostic, so the implementation
    uses the more conservative eps^(1/6) scale and never goes below the 
    requested floor.
    """
    eps = np.asarray(np.finfo(dtype).eps, dtype=dtype)
    return float(np.maximum(np.asarray(requested_step, dtype=dtype), np.power(eps, np.asarray(1.0 / 6.0, dtype=dtype), dtype=dtype)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-squared error between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def abs_quantile(x: np.ndarray, q: float) -> float:
    """q-quantile of absolute values, used for tail-size summaries."""
    x = np.asarray(x, dtype=float)
    return float(np.quantile(np.abs(x), q)) if x.size else float("nan")


def expected_shortfall_loss(pnl: np.ndarray, q: float = 0.95) -> float:
    """Expected shortfall of losses at level q.

    The hedging study works with P&L.  To report a downside-risk statistic, this
    function first converts P&L to loss = -P&L and then averages the worst tail.
    """
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return float("nan")
    loss = -pnl
    threshold = np.quantile(loss, q)
    tail = loss[loss >= threshold]
    return float(tail.mean()) if tail.size else float("nan")


# ============================================================================
# Input data loading and daily-state construction
# ----------------------------------------------------------------------------
# SEDS uses freely accessible local CSV files rather than live downloads.
# First the script reads raw SPX / FRED files, then it turns them into a daily
# state vector that later feeds the synthetic teacher panel.
# ============================================================================

def read_local_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file from disk and fail loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required local data file not found: {path}. "
            "Place the CSV in the same folder as the main program, or pass --data-dir."
        )
    return pd.read_csv(path)


def fetch_stooq_spx(data_dir: Path) -> pd.DataFrame:
    """Load the Stooq SPX history and standardize it to Date / SPX_Close."""
    path = data_dir / LOCAL_SOURCE_FILENAMES["spx"]
    df = read_local_csv(path)
    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path.name}. Expected at least Date and Close.")
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Close"] = numeric_series(out["Close"])
    out = out.rename(columns={"Close": "SPX_Close"}).dropna().sort_values("Date")
    if out.empty:
        raise ValueError(f"No usable rows were found in {path.name}.")
    return out.reset_index(drop=True)


def fetch_fred_series(series_id: str, data_dir: Path) -> pd.DataFrame:
    """Load one FRED-style CSV and rename it to a uniform Date / value layout."""
    path = data_dir / LOCAL_SOURCE_FILENAMES[series_id]
    df = read_local_csv(path)
    if len(df.columns) < 2:
        raise ValueError(f"Unexpected structure in {path.name}. Expected two columns.")
    df = df.iloc[:, :2].copy()
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df[series_id] = numeric_series(df[series_id])
    return df


def build_mock_daily_data(n_days: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic daily data set for dry runs and debugging.

    This is only a fallback/testing mode.  The real empirical exercise uses the
    local SPX and FRED CSV files.
    """
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
    return pd.DataFrame(
        {
            "Date": dates,
            "SPX_Close": spx,
            "DGS3MO": dgs3mo,
            "DGS2": dgs2,
            "DGS10": dgs10,
            "VIXCLS": vix,
            "VXVCLS": vxv,
        }
    )


def load_daily_market_data(use_mock: bool, n_mock_days: int, seed: int, data_dir: Path) -> pd.DataFrame:
    """Load the raw daily market data and add realized-volatility features.

    The output still contains the raw yield and implied-volatility proxy columns.
    A later step will forward-fill and trim the sample to the fully usable period.
    """
    if use_mock:
        spx_like = build_mock_daily_data(n_mock_days, seed).copy()
    else:
        spx_like = fetch_stooq_spx(data_dir).copy()

    spx_like = spx_like.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    spx_like["log_return"] = np.log(spx_like["SPX_Close"]).diff()
    spx_like["RV20"] = spx_like["log_return"].rolling(20, min_periods=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    spx_like["RV60"] = spx_like["log_return"].rolling(60, min_periods=60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    if use_mock:
        df = spx_like
    else:
        df = spx_like
        for sid in ["DGS3MO", "DGS2", "DGS10", "VIXCLS", "VXVCLS"]:
            fred = fetch_fred_series(sid, data_dir).sort_values("Date").drop_duplicates("Date")
            df = df.merge(fred, on="Date", how="left")
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def prepare_daily_state(df: pd.DataFrame) -> pd.DataFrame:
    """Build the daily state variables used by the teacher and the surrogates.

    Key transformations:
    - forward-fill the slower-moving FRED series,
    - convert VIX / VXV into IV30 / IV90 in decimal form,
    - drop the early rows where RV20 / RV60 are not yet available.
    """
    out = df.copy().sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    cols = ["DGS3MO", "DGS2", "DGS10", "VIXCLS", "VXVCLS"]
    out[cols] = out[cols].ffill()
    out["IV30"] = out["VIXCLS"] / 100.0
    out["IV90"] = out["VXVCLS"] / 100.0
    out = out.dropna(subset=["SPX_Close", "RV20", "RV60", "IV30", "IV90", "DGS3MO", "DGS2", "DGS10"]).reset_index(drop=True)
    return out


# ============================================================================
# Teacher-model ingredients
# ----------------------------------------------------------------------------
# The next block builds the synthetic teacher labels described in SEDS.
# Rates and implied vols are interpolated/extrapolated in maturity, then a
# simple blending rule forms sigma_teacher, and finally BSM generates the
# reference price / Delta / Gamma on the full fixed grid.
# ============================================================================

def piecewise_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Piecewise-linear interpolation with linear extrapolation at both ends."""
    x = as_float_array(x, dtype)
    xp = as_float_array(xp, dtype)
    fp = as_float_array(fp, dtype)
    y = np.interp(x, xp, fp)
    left = x < xp[0]
    if np.any(left):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left] = fp[0] + slope * (x[left] - xp[0])
    right = x > xp[-1]
    if np.any(right):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right] = fp[-1] + slope * (x[right] - xp[-1])
    return y


def alpha_tau(tau: np.ndarray) -> np.ndarray:
    """Short-maturity blend weight used in sigma_teacher.

    In the empirical subsection, shorter maturities place more weight on the
    implied-volatility proxy than longer maturities do.
    """
    tau = np.asarray(tau, dtype=float)
    return np.where(tau <= 30.0 / CALENDAR_DAYS_PER_YEAR, 0.8, 0.6)


def gamma_stabilized_target(S: np.ndarray, gamma: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Transform Gamma to the stabilized training target asinh(S^2 * Gamma).

    This makes the Gamma head numerically easier to train while preserving a
    clean invertible map back to the original Gamma scale.
    """
    S = as_float_array(S, dtype)
    gamma = as_float_array(gamma, dtype)
    return np.arcsinh((S ** np.asarray(2.0, dtype=dtype)) * gamma, dtype=dtype)


def gamma_from_stabilized(S: np.ndarray, gamma_tilde: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Invert the stabilized Gamma target back to ordinary Gamma."""
    S = as_float_array(S, dtype)
    gamma_tilde = as_float_array(gamma_tilde, dtype)
    tiny = np.asarray(1e-12, dtype=dtype)
    return np.sinh(gamma_tilde, dtype=dtype) / np.maximum(S ** np.asarray(2.0, dtype=dtype), tiny)


def bsm_call_price_delta_gamma(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standalone Black-Scholes-Merton call price, Delta, and Gamma engine.

    The implementation uses double precision by default, a stable vectorized
    normal CDF, and a log-ratio formula log(S)-log(K).
    """
    S = as_float_array(S, dtype)
    K = as_float_array(K, dtype)
    r = as_float_array(r, dtype)
    tau = np.maximum(as_float_array(tau, dtype), np.asarray(1e-8, dtype=dtype))
    sigma = np.maximum(as_float_array(sigma, dtype), np.asarray(1e-8, dtype=dtype))
    tiny = np.asarray(1e-12, dtype=dtype)
    sqrt_tau = np.sqrt(tau, dtype=dtype)
    log_ratio = np.log(np.maximum(S, tiny), dtype=dtype) - np.log(np.maximum(K, tiny), dtype=dtype)
    half = np.asarray(0.5, dtype=dtype)
    d1 = (log_ratio + (r + half * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    Nd1 = norm_cdf(d1, dtype=dtype)
    Nd2 = norm_cdf(d2, dtype=dtype)
    nd1 = norm_pdf(d1, dtype=dtype)
    discount = safe_exp(-r * tau, dtype=dtype)
    price = S * Nd1 - K * discount * Nd2
    delta = Nd1
    gamma = nd1 / np.maximum(S * sigma * sqrt_tau, tiny)
    return price, delta, gamma


def build_teacher_panel(daily: pd.DataFrame, dtype=np.float64) -> pd.DataFrame:
    """Expand the daily state into the full option panel used for learning.

    For each calendar date, the script evaluates the teacher on the fixed
    9 x 15 maturity-moneyness grid.  The resulting panel is the empirical
    supervised-learning data set for Models A, B, and C.
    """
    n_dates = len(daily)
    block = len(TAU_GRID_YEARS) * len(K_GRID)
    total = n_dates * block

    # Repeat each daily state across the full option grid for that date.
    date_rep = np.repeat(daily["Date"].values, block)
    S_rep = np.repeat(daily["SPX_Close"].values, block)
    rv20_rep = np.repeat(daily["RV20"].values, block)
    rv60_rep = np.repeat(daily["RV60"].values, block)
    iv30_rep = np.repeat(daily["IV30"].values, block)
    iv90_rep = np.repeat(daily["IV90"].values, block)
    tau_rep = np.tile(np.repeat(TAU_GRID_YEARS, len(K_GRID)), n_dates)
    k_rep = np.tile(np.tile(K_GRID, len(TAU_GRID_YEARS)), n_dates)

    # Preallocate arrays for the maturity-specific interpolated rate and IV.
    rates = np.empty(total, dtype=float)
    iv_tau = np.empty(total, dtype=float)
    tau_block = np.repeat(TAU_GRID_YEARS, len(K_GRID))
    rate_x = np.array([0.25, 2.0, 10.0], dtype=float)
    iv_x = np.array([30.0 / CALENDAR_DAYS_PER_YEAR, 90.0 / CALENDAR_DAYS_PER_YEAR], dtype=float)

    for i, row in daily.reset_index(drop=True).iterrows():
        # Fill one date-sized block at a time.  Each block contains all 9 x 15
        # contracts for that trading day.
        sl = slice(i * block, (i + 1) * block)
        rates[sl] = piecewise_linear_extrap(tau_block, rate_x, np.array([row["DGS3MO"], row["DGS2"], row["DGS10"]], dtype=dtype) / np.asarray(100.0, dtype=dtype), dtype=dtype)
        iv_tau[sl] = np.maximum(piecewise_linear_extrap(tau_block, iv_x, np.array([row["IV30"], row["IV90"]], dtype=dtype), dtype=dtype), np.asarray(1e-6, dtype=dtype))

    # Teacher volatility follows the empirical subsection's simple blend:
    # short maturities lean more heavily on IV, while longer maturities retain
    # a larger realized-volatility component.
    sigma_rep = alpha_tau(tau_rep) * iv_tau + (1.0 - alpha_tau(tau_rep)) * rv20_rep

    # The option grid is parameterized in log-moneyness k, so strike is rebuilt
    # via K = S * exp(k) before the teacher labels are generated.
    K_rep = S_rep * safe_exp(k_rep, dtype=dtype)
    price, delta, gamma = bsm_call_price_delta_gamma(S_rep, K_rep, rates, tau_rep, sigma_rep, dtype=dtype)

    panel = pd.DataFrame(
        {
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
            "sigma_teacher": sigma_rep,
            "price_teacher": price,
            "delta_teacher": delta,
            "gamma_teacher": gamma,
        }
    )
    # Mark the short-dated near-ATM subset that is emphasized in local metrics
    # and in Model C's boundary-aware weighting scheme.
    panel["boundary_region"] = ((panel["tau"] <= BOUNDARY_TAU_MAX) & (panel["k"].abs() <= BOUNDARY_K_MAX)).astype(int)
    return panel


def assign_splits(panel: pd.DataFrame, use_mock: bool) -> pd.DataFrame:
    """Assign train / validation / test splits by date.

    In the real-data workflow the split is calendar-based:
    - train through 2017-12-31
    - validation through 2021-12-31
    - test afterwards

    If that split would leave a bucket empty (or if mock data is used), the
    function falls back to a 60/20/20 split over unique dates.
    """
    out = panel.copy()
    if not use_mock:
        train_end = pd.Timestamp("2017-12-31")
        val_end = pd.Timestamp("2021-12-31")
        out["split"] = np.where(out["date"] <= train_end, "train", np.where(out["date"] <= val_end, "val", "test"))
        counts = out.groupby("split").size().to_dict()
        if min(counts.get("train", 0), counts.get("val", 0), counts.get("test", 0)) > 0:
            return out
    unique_dates = pd.Index(sorted(out["date"].unique()))
    n = len(unique_dates)
    train_cut = int(0.60 * n)
    val_cut = int(0.80 * n)
    mapper = {}
    for i, d in enumerate(unique_dates):
        mapper[d] = "train" if i < train_cut else ("val" if i < val_cut else "test")
    out["split"] = out["date"].map(mapper)
    return out


def maybe_subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Optionally subsample a split for faster experimentation.

    SEDS runs typically leave these caps at zero, meaning "use all rows".
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_values(["date", "tau", "k"]).reset_index(drop=True)


# ============================================================================
# Model containers, feature layout, and selection utilities
# ----------------------------------------------------------------------------
# The dataclass stores everything needed for later prediction and export:
# neural-network object, scalers, selected hyperparameters, and metadata about
# the validation-based candidate selection.
# ============================================================================

@dataclass
class TrainedModel:
    """Bundle the fitted model together with its scaling and selection metadata."""
    name: str
    model: MLPRegressor
    x_scaler: StandardScaler
    target_scaler: Optional[StandardScaler]
    mode: str  # 'A', 'B', 'C'
    lambda_delta: float
    lambda_gamma: float
    fd_k_step: float
    boundary_lambda: float = 0.0
    boundary_h: float = BOUNDARY_K_MAX
    selection_score: Optional[float] = None
    candidate_tag: str = ""
    dtype_name: str = "float64"


FEATURE_COLS = ["k", "tau", "r_tau", "RV20", "RV60", "IV30", "IV90", "IV_slope"]



def parse_float_grid(text: str) -> List[float]:
    """Parse a comma-separated grid specification from the command line."""
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("Expected a non-empty comma-separated float grid.")
    return vals


def safe_local_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the boundary subset, or the full frame if that subset is empty."""
    local = df[df["boundary_region"] == 1]
    return local if not local.empty else df


def metrics_from_prediction_frame(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    """Compute global and local RMSE metrics from one prediction frame."""
    pcol = f"price_{model_name}"
    dcol = f"delta_{model_name}"
    gcol = f"gamma_{model_name}"
    local = safe_local_subset(df)
    return {
        "global_price_rmse": rmse(df[pcol].values, df["price_teacher"].values),
        "global_delta_rmse": rmse(df[dcol].values, df["delta_teacher"].values),
        "global_gamma_rmse": rmse(df[gcol].values, df["gamma_teacher"].values),
        "local_price_rmse": rmse(local[pcol].values, local["price_teacher"].values),
        "local_delta_rmse": rmse(local[dcol].values, local["delta_teacher"].values),
        "local_gamma_rmse": rmse(local[gcol].values, local["gamma_teacher"].values),
    }


def composite_selection_score(candidate_metrics: Dict[str, float], baseline_metrics: Dict[str, float], args: argparse.Namespace) -> float:
    """Convenience wrapper using the default Model-B-style selection weights."""
    return composite_selection_score_with_weights(
        candidate_metrics,
        baseline_metrics,
        local_price_weight=args.selection_local_price_weight,
        global_price_weight=args.selection_global_price_weight,
        local_delta_weight=args.selection_local_delta_weight,
        local_gamma_weight=args.selection_local_gamma_weight,
    )


def composite_selection_score_with_weights(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    *,
    local_price_weight: float,
    global_price_weight: float,
    local_delta_weight: float,
    local_gamma_weight: float,
) -> float:
    """Score a candidate relative to the Model-A baseline.

    Lower is better.  The score is a weighted sum of candidate/baseline RMSE
    ratios, so values below 1 would indicate a weighted improvement over A.
    """
    eps = 1e-12
    return float(
        local_price_weight * (candidate_metrics["local_price_rmse"] / max(baseline_metrics["local_price_rmse"], eps))
        + global_price_weight * (candidate_metrics["global_price_rmse"] / max(baseline_metrics["global_price_rmse"], eps))
        + local_delta_weight * (candidate_metrics["local_delta_rmse"] / max(baseline_metrics["local_delta_rmse"], eps))
        + local_gamma_weight * (candidate_metrics["local_gamma_rmse"] / max(baseline_metrics["local_gamma_rmse"], eps))
    )


def fit_structured_candidate(
    name: str,
    train_df: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
    x_scaler: StandardScaler,
    lambda_delta: float,
    lambda_gamma: float,
    boundary_lambda: float = 0.0,
    boundary_h: Optional[float] = None,
) -> TrainedModel:
    """Fit one candidate structured model (Model B- or C-style).

    The output vector is
        [ price,
          sqrt(lambda_delta) * Delta,
          sqrt(lambda_gamma) * asinh(S^2 * Gamma) ].
    The sqrt-lambda scaling keeps the weighted-loss interpretation while letting
    the neural network train on an ordinary multi-output regression target.
    """
    X_train = x_scaler.transform(feature_matrix(train_df))

    # Stabilize Gamma before training and incorporate the Delta / Gamma weights
    # directly into the targets through sqrt(lambda) factors.
    dtype = resolve_dtype(args.dtype)
    gamma_train_tilde = gamma_stabilized_target(train_df["S"].values, train_df["gamma_teacher"].values, dtype=dtype)
    Y_train = np.column_stack(
        [
            train_df["price_teacher"].values,
            np.sqrt(lambda_delta) * train_df["delta_teacher"].values,
            np.sqrt(lambda_gamma) * gamma_train_tilde,
        ]
    )
    y_scaler = StandardScaler().fit(Y_train)
    Y_train_scaled = y_scaler.transform(Y_train)

    # Standardize the three output heads jointly before fitting the MLP.
    batch_size = min(max(1, args.batch_size), len(train_df))
    model = MLPRegressor(
        hidden_layer_sizes=(args.hidden_dim,) * args.depth,
        activation="relu",
        solver="adam",
        learning_rate_init=args.learning_rate,
        alpha=args.weight_decay,
        batch_size=batch_size,
        max_iter=args.max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=args.patience,
    )
    if boundary_lambda > 0.0:
        # Boundary-aware training: observations in the short-dated near-ATM
        # region receive larger sample weights, with smooth decay as |k| grows.
        effective_boundary_h = float(args.boundary_h if boundary_h is None else boundary_h)
        sw = boundary_weight(train_df, boundary_lambda, effective_boundary_h)
        model.fit(X_train, Y_train_scaled, sample_weight=sw)
    else:
        # Ordinary structured training with no boundary reweighting.
        model.fit(X_train, Y_train_scaled)
    return TrainedModel(
        name=name,
        model=model,
        x_scaler=x_scaler,
        target_scaler=y_scaler,
        mode=name.split("_")[-1],
        lambda_delta=lambda_delta,
        lambda_gamma=lambda_gamma,
        fd_k_step=args.fd_k_step,
        boundary_lambda=boundary_lambda,
        boundary_h=float(args.boundary_h if boundary_h is None else boundary_h),
        dtype_name=args.dtype,
    )


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the model feature matrix in the fixed column order."""
    return df[FEATURE_COLS].values.astype(float)


def boundary_weight(df: pd.DataFrame, lambda_boundary: float, boundary_h: float) -> np.ndarray:
    """Smooth sample-weight bump for the boundary region.

    The indicator keeps the extra weight within the short-maturity strip, while
    the Gaussian-looking k-term concentrates the emphasis near the money.
    """
    ind = (df["tau"].values <= BOUNDARY_TAU_MAX).astype(float)
    return 1.0 + lambda_boundary * np.exp(-(df["k"].values ** 2) / (boundary_h ** 2)) * ind



# ============================================================================
# Model fitting and validation-based candidate selection
# ----------------------------------------------------------------------------
# Model A is a plain price-only network.
# Model B is a structured multi-output network selected over a small Delta/Gamma
# grid without boundary weighting.
# Model C uses the same structured architecture but adds boundary weighting and
# a wider, C-specific rebalance search intended to improve local price fit.
# ============================================================================

def fit_models(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int, args: argparse.Namespace) -> Dict[str, TrainedModel]:
    """Fit all three surrogate models and return the selected versions.

    Workflow:
    1. Fit Model A on price only.
    2. Use Model A's validation metrics as the baseline for structured-model
       selection.
    3. Search a small candidate grid for Model B.
    4. Search a broader, rebalanced candidate grid for Model C.
    """
    x_scaler = StandardScaler().fit(feature_matrix(train_df))
    X_train = x_scaler.transform(feature_matrix(train_df))
    batch_size = min(max(1, args.batch_size), len(train_df))

    # ----------------------------
    # Model A: price-only baseline.
    # ----------------------------
    model_A = MLPRegressor(
        hidden_layer_sizes=(args.hidden_dim,) * args.depth,
        activation="relu",
        solver="adam",
        learning_rate_init=args.learning_rate,
        alpha=args.weight_decay,
        batch_size=batch_size,
        max_iter=args.max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=args.patience,
    )
    model_A.fit(X_train, train_df["price_teacher"].values)
    trained_A = TrainedModel("Model_A", model_A, x_scaler, None, "A", args.lambda_delta, args.lambda_gamma, args.fd_k_step, dtype_name=args.dtype)

    # Model A provides the baseline metric vector that B and C are judged
    # against on the validation set.
    baseline_metrics = None
    if not val_df.empty:
        baseline_pred = prediction_frame(trained_A, val_df)
        baseline_metrics = metrics_from_prediction_frame(baseline_pred, "Model_A")

    def select_structured_model(name: str, base_seed: int, use_boundary: bool) -> TrainedModel:
        """Search the candidate grid and keep the best structured model."""
        is_model_c = (name == "Model_C")
        if is_model_c:
            # Model C gets its own broader, softer rebalance search.
            delta_grid = sorted(set(max(1e-8, args.lambda_delta * m) for m in parse_float_grid(args.selection_c_delta_multipliers)))
            gamma_grid = sorted(set(max(1e-8, args.lambda_gamma * m) for m in parse_float_grid(args.selection_c_gamma_multipliers)))
            boundary_grid = sorted(set(max(0.0, args.lambda_boundary * m) for m in parse_float_grid(args.selection_c_boundary_multipliers))) if use_boundary else [0.0]
            boundary_h_grid = sorted(set(max(1e-6, args.boundary_h * m) for m in parse_float_grid(args.selection_c_boundary_h_multipliers))) if use_boundary else [args.boundary_h]
            local_price_weight = args.selection_c_local_price_weight
            global_price_weight = args.selection_c_global_price_weight
            local_delta_weight = args.selection_c_local_delta_weight
            local_gamma_weight = args.selection_c_local_gamma_weight
        else:
            # Model B keeps the ordinary structured-model grid and selection weights.
            delta_grid = sorted(set(max(1e-8, args.lambda_delta * m) for m in parse_float_grid(args.selection_delta_multipliers)))
            gamma_grid = sorted(set(max(1e-8, args.lambda_gamma * m) for m in parse_float_grid(args.selection_gamma_multipliers)))
            boundary_grid = [0.0]
            if use_boundary:
                boundary_grid = sorted(set(max(0.0, args.lambda_boundary * m) for m in parse_float_grid(args.selection_boundary_multipliers)))
            boundary_h_grid = [args.boundary_h]
            local_price_weight = args.selection_local_price_weight
            global_price_weight = args.selection_global_price_weight
            local_delta_weight = args.selection_local_delta_weight
            local_gamma_weight = args.selection_local_gamma_weight

        best_model: Optional[TrainedModel] = None
        best_score = float("inf")

        # Brute-force grid search over the candidate tuples.  Each tuple
        # corresponds to one full MLP fit on the training set.
        for idx, (ld, lg, lb, bh) in enumerate(product(delta_grid, gamma_grid, boundary_grid, boundary_h_grid)):
            candidate = fit_structured_candidate(
                name=name,
                train_df=train_df,
                seed=base_seed + idx,
                args=args,
                x_scaler=x_scaler,
                lambda_delta=ld,
                lambda_gamma=lg,
                boundary_lambda=lb if use_boundary else 0.0,
                boundary_h=bh,
            )
            if val_df.empty or baseline_metrics is None:
                candidate.selection_score = float("nan")
                candidate.candidate_tag = (
                    f"lambda_delta={ld:.6g}|lambda_gamma={lg:.6g}|"
                    f"lambda_boundary={lb:.6g}|boundary_h={bh:.6g}"
                )
                return candidate

            # Evaluate this candidate on the validation set relative to the
            # Model-A baseline.  Lower composite score is better.
            pred_val = prediction_frame(candidate, val_df)
            cand_metrics = metrics_from_prediction_frame(pred_val, name)
            score = composite_selection_score_with_weights(
                cand_metrics,
                baseline_metrics,
                local_price_weight=local_price_weight,
                global_price_weight=global_price_weight,
                local_delta_weight=local_delta_weight,
                local_gamma_weight=local_gamma_weight,
            )
            candidate.selection_score = score
            candidate.candidate_tag = (
                f"lambda_delta={ld:.6g}|lambda_gamma={lg:.6g}|"
                f"lambda_boundary={lb:.6g}|boundary_h={bh:.6g}"
            )

            if score < best_score:
                best_score = score
                best_model = candidate

        assert best_model is not None
        return best_model

    # Final selected structured models.
    trained_B = select_structured_model("Model_B", seed + 100, use_boundary=False)
    trained_C = select_structured_model("Model_C", seed + 200, use_boundary=True)

    return {
        "Model_A": trained_A,
        "Model_B": trained_B,
        "Model_C": trained_C,
    }


# ============================================================================
# Prediction helpers and test-set metric tables
# ============================================================================

def predict_price_only(trained: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Predict only prices, used directly by Model A and internally elsewhere."""
    X = trained.x_scaler.transform(feature_matrix(df))
    return trained.model.predict(X).reshape(-1)


def predict_price_delta_gamma(trained: TrainedModel, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict price, Delta, and Gamma from a fitted surrogate.

    Model A only learns prices, so its Greeks are reconstructed by finite
    differences in k and then mapped back to S-derivatives.

    Models B and C predict three heads directly.  Their Delta and Gamma outputs
    are then unscaled and Gamma is inverted from the stabilized target.
    """
    if trained.mode == "A":
        price0 = predict_price_only(trained, df)

        # Finite-difference reconstruction for Model A:
        #   c_k  = dC/dk
        #   c_kk = d^2C/dk^2
        # then use the k-to-S identities implied by K = S exp(k).
        #
        # A fourth-order centered stencil is used rather than the simpler
        # three-point formula.  This lowers truncation error, while the adaptive
        # h floor avoids making the second derivative dominated by roundoff.
        dtype = resolve_dtype(trained.dtype_name)
        h = fd_base_step(dtype=dtype, requested_step=trained.fd_k_step)
        df_p1 = df.copy(); df_p1["k"] = df_p1["k"] + h; df_p1["IV_slope"] = df_p1["IV90"] - df_p1["IV30"]
        df_m1 = df.copy(); df_m1["k"] = df_m1["k"] - h; df_m1["IV_slope"] = df_m1["IV90"] - df_m1["IV30"]
        df_p2 = df.copy(); df_p2["k"] = df_p2["k"] + 2.0 * h; df_p2["IV_slope"] = df_p2["IV90"] - df_p2["IV30"]
        df_m2 = df.copy(); df_m2["k"] = df_m2["k"] - 2.0 * h; df_m2["IV_slope"] = df_m2["IV90"] - df_m2["IV30"]
        price_p1 = predict_price_only(trained, df_p1)
        price_m1 = predict_price_only(trained, df_m1)
        price_p2 = predict_price_only(trained, df_p2)
        price_m2 = predict_price_only(trained, df_m2)
        c_k = (-price_p2 + 8.0 * price_p1 - 8.0 * price_m1 + price_m2) / (12.0 * h)
        c_kk = (-price_p2 + 16.0 * price_p1 - 30.0 * price0 + 16.0 * price_m1 - price_m2) / (12.0 * h * h)
        S = df["S"].values
        delta = -c_k / np.maximum(S, 1e-8)
        gamma = (c_kk + c_k) / np.maximum(S ** 2, 1e-8)
        return price0, delta, gamma

    X = trained.x_scaler.transform(feature_matrix(df))
    pred_scaled = trained.model.predict(X)
    pred = trained.target_scaler.inverse_transform(pred_scaled)
    price = pred[:, 0]

    # Undo the sqrt(lambda) weighting baked into the training targets.
    delta = pred[:, 1] / np.sqrt(trained.lambda_delta)
    gamma_tilde = pred[:, 2] / np.sqrt(trained.lambda_gamma)
    gamma = gamma_from_stabilized(df["S"].values, gamma_tilde, dtype=resolve_dtype(trained.dtype_name))
    return price, delta, gamma


def prediction_frame(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Attach one model's predictions to the teacher labels for comparison."""
    price, delta, gamma = predict_price_delta_gamma(trained, df)
    out = df[["date", "S", "K", "k", "tau", "price_teacher", "delta_teacher", "gamma_teacher", "boundary_region"]].copy()
    out[f"price_{trained.name}"] = price
    out[f"delta_{trained.name}"] = delta
    out[f"gamma_{trained.name}"] = gamma
    return out


def build_metrics_tables(pred_frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the global and local RMSE tables reported in SEDS."""
    global_rows, local_rows = [], []
    for name, df in pred_frames.items():
        pcol, dcol, gcol = f"price_{name}", f"delta_{name}", f"gamma_{name}"
        global_rows.append({
            "model": name,
            "price_rmse": rmse(df[pcol].values, df["price_teacher"].values),
            "delta_rmse": rmse(df[dcol].values, df["delta_teacher"].values),
            "gamma_rmse": rmse(df[gcol].values, df["gamma_teacher"].values),
        })
        local = df[df["boundary_region"] == 1]
        local_rows.append({
            "model": name,
            "price_rmse": rmse(local[pcol].values, local["price_teacher"].values),
            "delta_rmse": rmse(local[dcol].values, local["delta_teacher"].values),
            "gamma_rmse": rmse(local[gcol].values, local["gamma_teacher"].values),
        })
    return pd.DataFrame(global_rows), pd.DataFrame(local_rows)


# ============================================================================
# Five-day local hedging backtest
# ----------------------------------------------------------------------------
# SEDS also evaluates the surrogates through a simple discrete hedging
# experiment on local-boundary contracts.  The following helpers rebuild the
# evolving state of each contract and summarize model-specific P&L.
# ============================================================================

def daily_lookup(daily: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
    """Create a fast date -> state dictionary for the hedging loop."""
    lookup: Dict[pd.Timestamp, Dict[str, float]] = {}
    for _, row in daily.iterrows():
        lookup[pd.Timestamp(row["Date"])] = {
            "S": float(row["SPX_Close"]),
            "RV20": float(row["RV20"]),
            "RV60": float(row["RV60"]),
            "IV30": float(row["IV30"]),
            "IV90": float(row["IV90"]),
            "DGS3MO": float(row["DGS3MO"]) / 100.0,
            "DGS2": float(row["DGS2"]) / 100.0,
            "DGS10": float(row["DGS10"]) / 100.0,
        }
    return lookup


def remaining_tau(start_date: pd.Timestamp, current_date: pd.Timestamp, tau_init: float) -> float:
    """Compute remaining time to maturity in years."""
    return max(tau_init - (current_date - start_date).days / CALENDAR_DAYS_PER_YEAR, 0.0)


def contract_state(date: pd.Timestamp, K: float, tau_rem: float, state: Dict[str, float]) -> Dict[str, float]:
    """Rebuild one contract's feature vector at a given rebalance date."""
    k = math.log(max(K, 1e-12) / max(state["S"], 1e-12))
    r_tau = float(piecewise_linear_extrap(np.array([tau_rem]), np.array([0.25, 2.0, 10.0]), np.array([state["DGS3MO"], state["DGS2"], state["DGS10"]]))[0])
    iv_tau = float(np.maximum(piecewise_linear_extrap(np.array([tau_rem]), np.array([30.0 / CALENDAR_DAYS_PER_YEAR, 90.0 / CALENDAR_DAYS_PER_YEAR]), np.array([state["IV30"], state["IV90"]]))[0], 1e-6))
    sigma = float(alpha_tau(np.array([tau_rem]))[0] * iv_tau + (1.0 - alpha_tau(np.array([tau_rem]))[0]) * state["RV20"])
    return {
        "date": date,
        "S": state["S"],
        "K": K,
        "k": k,
        "tau": tau_rem,
        "r_tau": r_tau,
        "RV20": state["RV20"],
        "RV60": state["RV60"],
        "IV30": state["IV30"],
        "IV90": state["IV90"],
        "IV_slope": state["IV90"] - state["IV30"],
        "sigma_teacher": sigma,
    }


def predict_single(trained: TrainedModel, row: Dict[str, float]) -> Tuple[float, float, float]:
    """Small wrapper so the hedging loop can score one contract state at a time."""
    df = pd.DataFrame([row])
    return tuple(float(x[0]) for x in predict_price_delta_gamma(trained, df))


def teacher_price(row: Dict[str, float]) -> float:
    """Teacher option value at a single state, with intrinsic value at expiry."""
    if row["tau"] <= 0.0:
        return max(row["S"] - row["K"], 0.0)
    price, _, _ = bsm_call_price_delta_gamma(np.array([row["S"]]), np.array([row["K"]]), np.array([row["r_tau"]]), np.array([row["tau"]]), np.array([row["sigma_teacher"]]))
    return float(price[0])


def hedging_metrics(test_df: pd.DataFrame, daily: pd.DataFrame, models: Dict[str, TrainedModel], max_paths: int = 0) -> pd.DataFrame:
    """Run the five-day discrete hedging experiment for the local region.

    For each eligible boundary-region test contract:
    - initialize the hedge using the model's price and Delta,
    - rebalance Delta once per trading day for five days,
    - compare final hedge value to the teacher option value,
    - aggregate the resulting P&L distribution by model.
    """
    daily = daily.sort_values("Date").reset_index(drop=True)
    pos = {pd.Timestamp(d): i for i, d in enumerate(daily["Date"])}
    lookup = daily_lookup(daily)
    candidates = test_df[test_df["boundary_region"] == 1].copy()

    # Keep only contracts for which a full 5-trading-day hedge window exists
    # and the option remains alive at the end of that window.
    keep = []
    for _, row in candidates.iterrows():
        p = pos.get(pd.Timestamp(row["date"]))
        if p is None or p + 5 >= len(daily):
            keep.append(False)
            continue
        end_date = pd.Timestamp(daily.iloc[p + 5]["Date"])
        keep.append(remaining_tau(pd.Timestamp(row["date"]), end_date, float(row["tau"])) > 0.0)
    candidates = candidates.loc[np.array(keep, dtype=bool)].reset_index(drop=True)

    if max_paths > 0 and len(candidates) > max_paths:
        candidates = candidates.sample(n=max_paths, random_state=123).sort_values(["date", "tau", "k"]).reset_index(drop=True)

    rows = []
    for _, contract in candidates.iterrows():
        # Each contract is hedged separately under each model so that the P&L
        # distribution is directly comparable across A, B, and C.
        start_date = pd.Timestamp(contract["date"])
        p0 = pos[start_date]
        K = float(contract["K"])
        tau0 = float(contract["tau"])
        for name, trained in models.items():
            st0 = contract_state(start_date, K, tau0, lookup[start_date])
            price0, delta0, _ = predict_single(trained, st0)

            # Start from the model-implied option value and Delta hedge:
            # portfolio = Delta * S + cash = option price.
            cash = price0 - delta0 * st0["S"]
            delta_prev = delta0
            prev_date = start_date
            for step in range(1, 6):
                # Advance one trading day, accrue cash at the interpolated
                # short rate, then rebalance to the new model Delta.
                curr_date = pd.Timestamp(daily.iloc[p0 + step]["Date"])
                tau_rem = remaining_tau(start_date, curr_date, tau0)
                st = contract_state(curr_date, K, tau_rem, lookup[curr_date])
                dt = max((curr_date - prev_date).days / CALENDAR_DAYS_PER_YEAR, 0.0)
                cash = cash * math.exp(st["r_tau"] * dt)
                port = delta_prev * st["S"] + cash
                _, delta_new, _ = predict_single(trained, st)
                cash = port - delta_new * st["S"]
                delta_prev = delta_new
                prev_date = curr_date
            final_date = pd.Timestamp(daily.iloc[p0 + 5]["Date"])
            final_tau = remaining_tau(start_date, final_date, tau0)
            stf = contract_state(final_date, K, final_tau, lookup[final_date])
            # Final hedge error: hedge portfolio minus teacher option value.
            cash = cash * math.exp(stf["r_tau"] * max((final_date - prev_date).days / CALENDAR_DAYS_PER_YEAR, 0.0))
            pnl = delta_prev * stf["S"] + cash - teacher_price(stf)
            rows.append({"model": name, "pnl": pnl})

    if not rows:
        return pd.DataFrame(
            [
                {
                    "model": name,
                    "mean_pnl": float("nan"),
                    "std_pnl": float("nan"),
                    "abs_pnl_q95": float("nan"),
                    "es95_loss": float("nan"),
                    "n_contracts": 0,
                }
                for name in sorted(models.keys())
            ]
        )

    out = []
    hedge = pd.DataFrame(rows)
    for name in sorted(models.keys()):
        frag = hedge[hedge["model"] == name]
        if frag.empty:
            out.append({
                "model": name,
                "mean_pnl": float("nan"),
                "std_pnl": float("nan"),
                "abs_pnl_q95": float("nan"),
                "es95_loss": float("nan"),
                "n_contracts": 0,
            })
            continue
        pnl = frag["pnl"].values
        out.append({
            "model": name,
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl, ddof=0)),
            "abs_pnl_q95": abs_quantile(pnl, 0.95),
            "es95_loss": expected_shortfall_loss(pnl, 0.95),
            "n_contracts": int(len(frag)),
        })
    return pd.DataFrame(out).sort_values("model").reset_index(drop=True)


def sample_count_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Count observations by split and by local-boundary membership."""
    rows = []
    for split, frag in panel.groupby("split"):
        rows.append({
            "split": split,
            "n_samples": int(len(frag)),
            "n_boundary_samples": int(frag["boundary_region"].sum()),
            "n_dates": int(frag["date"].nunique()),
        })
    return pd.DataFrame(rows)



def numerical_diagnostics_table(args: argparse.Namespace) -> pd.DataFrame:
    """Record numerical-analysis choices used in the empirical run."""
    dtype = resolve_dtype(args.dtype)
    return pd.DataFrame([
        {"item": "floating_dtype", "value": args.dtype, "comment": "Teacher labels and finite-difference diagnostics use this dtype."},
        {"item": "machine_epsilon", "value": f"{machine_epsilon(dtype):.18e}", "comment": "Roundoff scale for the selected dtype."},
        {"item": "model_a_fd_order", "value": "4", "comment": "Fourth-order centered stencil in log-moneyness k."},
        {"item": "model_a_requested_fd_step", "value": f"{args.fd_k_step:.18e}", "comment": "Requested finite-difference floor."},
        {"item": "model_a_effective_fd_step", "value": f"{fd_base_step(dtype, args.fd_k_step):.18e}", "comment": "max(requested step, eps^(1/6)) for Gamma stability."},
        {"item": "normal_cdf_backend", "value": "scipy.special.ndtr" if _normal_cdf is not None else "math.erf fallback", "comment": "Vectorized stable normal CDF for BSM labels."},
        {"item": "gamma_target", "value": "asinh(S^2 * gamma)", "comment": "Scale-stabilized Gamma target for Models B and C."},
    ])

def write_table(df: pd.DataFrame, csv_path: Path) -> None:
    """Write one table as CSV file."""
    df.to_csv(csv_path, index=False)


# ============================================================================
# Command-line arguments and main program
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Define command-line controls for data, training, selection, and output."""
    p = argparse.ArgumentParser(description="Implement the SEDS free-data empirical example.")
    p.add_argument("--output-dir", type=str, default="SEDS_empirical_outputs")
    p.add_argument("--mock-data", action="store_true")
    p.add_argument("--data-dir", type=str, default=None, help="Directory containing spx_stooq.csv and the five FRED CSV files. Defaults to the script folder.")
    p.add_argument("--n-mock-days", type=int, default=1200)
    # Core MLP training controls shared by the three models.
    p.add_argument("--max-iter", type=int, default=120)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=10)
    # Baseline structured-model weights and boundary emphasis.
    p.add_argument("--lambda-delta", type=float, default=1.0)
    p.add_argument("--lambda-gamma", type=float, default=5.0)
    p.add_argument("--lambda-boundary", type=float, default=4.0)
    p.add_argument("--boundary-h", type=float, default=0.02)
    # Validation-selection weights for Model B.
    p.add_argument("--selection-local-price-weight", type=float, default=3.0)
    p.add_argument("--selection-global-price-weight", type=float, default=0.75)
    p.add_argument("--selection-local-delta-weight", type=float, default=1.25)
    p.add_argument("--selection-local-gamma-weight", type=float, default=1.0)
    p.add_argument("--selection-delta-multipliers", type=str, default="1.0,0.85")
    p.add_argument("--selection-gamma-multipliers", type=str, default="1.0,0.7,0.5")
    p.add_argument("--selection-boundary-multipliers", type=str, default="1.0,0.7,0.5")
    # C-specific rebalance search: both the candidate grids and the score
    # weights can differ from Model B.
    p.add_argument("--selection-c-local-price-weight", type=float, default=4.5)
    p.add_argument("--selection-c-global-price-weight", type=float, default=1.0)
    p.add_argument("--selection-c-local-delta-weight", type=float, default=1.0)
    p.add_argument("--selection-c-local-gamma-weight", type=float, default=0.75)
    p.add_argument("--selection-c-delta-multipliers", type=str, default="0.85,1.0")
    p.add_argument("--selection-c-gamma-multipliers", type=str, default="0.4,0.55,0.7,0.85")
    p.add_argument("--selection-c-boundary-multipliers", type=str, default="0.2,0.35,0.5,0.7")
    p.add_argument("--selection-c-boundary-h-multipliers", type=str, default="1.0,1.25,1.5")
    # Miscellaneous runtime controls.
    p.add_argument("--fd-k-step", type=float, default=1e-3)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--max-hedge-paths", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, choices=["float64", "float32"], default="float64", help="Floating dtype for teacher labels and numerical diagnostics. Paper tables use float64.")
    return p.parse_args()


def main() -> None:
    """Run the full SEDS empirical pipeline from raw files to final tables."""
    args = parse_args()
    set_seed(args.seed)

    # Create the output folder structure.
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "data"
    table_dir = out_dir / "tables"
    model_dir = out_dir / "models"
    ensure_dir(data_dir); ensure_dir(table_dir); ensure_dir(model_dir)

    # Step 1. Load raw data and build the cleaned daily state data set.
    data_dir_input = Path(args.data_dir) if args.data_dir else default_data_dir()
    daily = prepare_daily_state(load_daily_market_data(args.mock_data, args.n_mock_days, args.seed, data_dir_input))
    daily.to_csv(data_dir / "daily_market_state.csv", index=False)

    # Step 2. Expand the daily state into the full teacher-labeled option panel
    # and assign train / validation / test splits.
    dtype = resolve_dtype(args.dtype)
    panel = assign_splits(build_teacher_panel(daily, dtype=dtype), use_mock=args.mock_data)
    panel.to_csv(data_dir / "teacher_panel_full.csv", index=False)

    # Step 3. Optionally subsample for faster experimentation, then fit the
    # three surrogate models.
    train_df = maybe_subsample(panel[panel["split"] == "train"], args.max_train_samples, args.seed)
    val_df = maybe_subsample(panel[panel["split"] == "val"], args.max_val_samples, args.seed + 1)
    test_df = maybe_subsample(panel[panel["split"] == "test"], args.max_test_samples, args.seed + 2)

    models = fit_models(train_df, val_df, args.seed, args)

    # Step 4. Score each selected model on the test set and save the fitted
    # objects / metadata for later reuse.
    pred_frames = {}
    for name, trained in models.items():
        pred = prediction_frame(trained, test_df)
        pred.to_csv(data_dir / f"predictions_{name}_test.csv", index=False)
        pred_frames[name] = pred
        np.savez(
            model_dir / f"{name}.npz",
            x_mean=trained.x_scaler.mean_,
            x_scale=trained.x_scaler.scale_,
            fd_k_step=np.array([trained.fd_k_step]),
            lambda_delta=np.array([trained.lambda_delta]),
            lambda_gamma=np.array([trained.lambda_gamma]),
            gamma_target_transform=np.array(["asinh(S^2*gamma)"], dtype=object),
            boundary_lambda=np.array([trained.boundary_lambda]),
            boundary_h=np.array([trained.boundary_h]),
            selection_score=np.array([np.nan if trained.selection_score is None else trained.selection_score]),
            candidate_tag=np.array([trained.candidate_tag], dtype=object),
        )
        with open(model_dir / f"{name}.pkl", "wb") as fh:
            pickle.dump(trained, fh)

    # Step 5. Build the paper-facing summary tables as CSV files, including the hedging test.
    table_counts = sample_count_table(panel)
    global_metrics, local_metrics = build_metrics_tables(pred_frames)
    hedge = hedging_metrics(test_df, daily, models, max_paths=args.max_hedge_paths)
    numerical_diagnostics = numerical_diagnostics_table(args)

    write_table(table_counts, table_dir / "table_sample_counts.csv")
    write_table(global_metrics, table_dir / "table_global_metrics.csv")
    write_table(local_metrics, table_dir / "table_local_boundary_metrics.csv")
    write_table(hedge, table_dir / "table_hedging_metrics.csv")
    write_table(numerical_diagnostics, table_dir / "table_numerical_diagnostics.csv")

    # Step 6. Write a manifest so the run is self-documenting.
    manifest = {
        "data_files": [
            str(data_dir / "daily_market_state.csv"),
            str(data_dir / "teacher_panel_full.csv"),
            str(data_dir / "predictions_Model_A_test.csv"),
            str(data_dir / "predictions_Model_B_test.csv"),
            str(data_dir / "predictions_Model_C_test.csv"),
        ],
        "table_files": [
            str(table_dir / "table_sample_counts.csv"),
            str(table_dir / "table_global_metrics.csv"),
            str(table_dir / "table_local_boundary_metrics.csv"),
            str(table_dir / "table_hedging_metrics.csv"),
            str(table_dir / "table_numerical_diagnostics.csv"),
        ],
        "notes": [
            "Teacher labels are generated with a standalone BSE-style BSM call engine.",
            "Dividend yield q is fixed at 0.0 to match the subsection design.",
            "Model A is price-only; Models B and C use weighted multi-output Greek supervision.",
            "The Gamma head is trained on asinh(S^2 gamma) and inverted back at prediction time.",
            "Model A Greeks use a fourth-order centered stencil in log-moneyness with an epsilon-based step floor for Gamma stability.",
            "The BSM teacher uses a stable vectorized normal CDF and log-ratio d1 construction.",
            "Models B and C are selected on validation scores that explicitly emphasize local boundary-region price RMSE while retaining local Delta and Gamma objectives.",
            "Model C uses a separate rebalanced candidate search with softer boundary/gamma settings and a more price-aware selector; Model B is unchanged.",
        ],
        "data_dir_used": str(data_dir_input.resolve()),
        "expected_local_files": list(LOCAL_SOURCE_FILENAMES.values()),
        "args": vars(args),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Finished. Output directory:", out_dir)


if __name__ == "__main__":
    main()
