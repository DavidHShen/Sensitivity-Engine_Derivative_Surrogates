# Input data

The empirical script expects six local CSV files in this directory when run with `--data-dir data/raw`.

| Filename | Source used in the manuscript | Required structure |
|---|---|---|
| `spx_stooq.csv` | Stooq S&P 500 historical series | Columns `Date` and `Close` |
| `DGS3MO.csv` | FRED DGS3MO | Date column followed by the series value |
| `DGS2.csv` | FRED DGS2 | Date column followed by the series value |
| `DGS10.csv` | FRED DGS10 | Date column followed by the series value |
| `VIXCLS.csv` | FRED VIXCLS | Date column followed by the series value |
| `VXVCLS.csv` | FRED VXVCLS | Date column followed by the series value |

The files may contain missing source observations represented as blanks or nonnumeric markers; the loader converts values to numeric form, removes unusable source rows, and performs backward as-of alignment subject to the configured source-age limit.

## Validation

```bash
python scripts/validate_inputs.py \
  --data-dir data/raw \
  --write-manifest data/input_manifest.csv
```

The validator records file size, SHA-256, usable row count, date range, duplicate-date count, and basic positivity checks.

## Redistribution

Raw third-party series are not included. Download access does not necessarily confer redistribution rights. Users must obtain the data under the applicable provider terms.

## Exact numerical reproduction

A current provider download may not match the historical vintage used for the manuscript. Exact reproduction requires the same six input files or files independently verified to be identical by SHA-256.
