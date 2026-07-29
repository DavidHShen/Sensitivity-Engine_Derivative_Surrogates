from __future__ import annotations

"""Validate the six local empirical input files and write a checksum manifest."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

EXPECTED = {
    "spx_stooq.csv": {"required": {"Date", "Close"}, "positive": True},
    "DGS3MO.csv": {"series": "DGS3MO", "positive": False},
    "DGS2.csv": {"series": "DGS2", "positive": False},
    "DGS10.csv": {"series": "DGS10", "positive": False},
    "VIXCLS.csv": {"series": "VIXCLS", "positive": True},
    "VXVCLS.csv": {"series": "VXVCLS", "positive": True},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def inspect_file(path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "filename": path.name,
        "exists": path.exists(),
        "valid": False,
        "sha256": "",
        "file_size_bytes": "",
        "raw_rows": 0,
        "usable_rows": 0,
        "start_date": "",
        "end_date": "",
        "duplicate_dates": "",
        "nonpositive_values": "",
        "value_column": "",
        "message": "",
    }
    if not path.exists():
        row["message"] = "missing"
        return row

    row["sha256"] = sha256_file(path)
    row["file_size_bytes"] = path.stat().st_size

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        row["message"] = f"CSV read failed: {exc}"
        return row

    row["raw_rows"] = len(frame)
    if path.name == "spx_stooq.csv":
        required = spec["required"]
        missing = sorted(required.difference(frame.columns))
        if missing:
            row["message"] = f"missing columns: {missing}"
            return row
        date_col = "Date"
        value_col = "Close"
    else:
        if len(frame.columns) < 2:
            row["message"] = "requires at least two columns"
            return row
        date_col = frame.columns[0]
        preferred = spec["series"]
        value_col = preferred if preferred in frame.columns else frame.columns[1]

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    values = numeric(frame[value_col])
    usable = pd.DataFrame({"Date": dates, "Value": values}).dropna()
    usable = usable[np.isfinite(usable["Value"])]

    row["value_column"] = str(value_col)
    row["usable_rows"] = len(usable)
    row["duplicate_dates"] = int(usable["Date"].duplicated().sum())
    row["nonpositive_values"] = int((usable["Value"] <= 0.0).sum())
    if not usable.empty:
        row["start_date"] = usable["Date"].min().date().isoformat()
        row["end_date"] = usable["Date"].max().date().isoformat()

    problems: List[str] = []
    if usable.empty:
        problems.append("no usable date/value rows")
    if row["duplicate_dates"]:
        problems.append(f"{row['duplicate_dates']} duplicate usable dates")
    if spec.get("positive") and row["nonpositive_values"]:
        problems.append(f"{row['nonpositive_values']} nonpositive values")

    row["valid"] = not problems
    row["message"] = "ok" if not problems else "; ".join(problems)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--write-manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    rows = [inspect_file(data_dir / name, spec) for name, spec in EXPECTED.items()]
    frame = pd.DataFrame(rows)

    display_cols = [
        "filename", "exists", "valid", "usable_rows", "start_date",
        "end_date", "duplicate_dates", "nonpositive_values", "message",
    ]
    print(frame[display_cols].to_string(index=False))

    if args.write_manifest:
        output = Path(args.write_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"\nManifest written to {output}")

    return 0 if bool(frame["valid"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
