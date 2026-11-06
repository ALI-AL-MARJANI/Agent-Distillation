"""Download UCI Online Retail II (CC BY 4.0, DOI 10.24432/C5CG6D), check its sha256, convert it to one CSV.

python scripts/get_data.py          # ~25 s once downloaded (the UCI download itself can take ~15 min)
python scripts/get_data.py --zip ../Data-Analyst-Agent/data/raw/online_retail_ii.zip   # reuse a local copy

Copied from Data-Analyst-Agent (same pinned sha256, same CSV), with one addition: --zip.

The .xlsx has two sheets ("Year 2009-2010", "Year 2010-2011") that overlap: both contain the invoices of
2010-12-01 to 2010-12-09 (22,523 identical rows). The second copy is dropped. Nothing else is cleaned,
because the benchmark questions are about the messy parts (cancellations, missing Customer ID, ...).
Writes data/raw/online_retail_ii.csv and data/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from distill.runinfo import REPO_ROOT, write_json

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
RAW = REPO_ROOT / "data/raw"
CSV = RAW / "online_retail_ii.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(local_zip: str | None = None) -> bytes:
    zip_path = RAW / "online_retail_ii.zip"
    if local_zip:  # read in place, not copied (saves 45 MB)
        zip_path = Path(local_zip)
        data = zip_path.read_bytes()
    elif zip_path.exists():
        data = zip_path.read_bytes()
    else:
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r:
            data = r.read()
        RAW.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(data)
    digest = sha256(data)
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: got {digest}, expected {SHA256}. Delete {zip_path} and retry.")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", help="path to an already downloaded online_retail_ii.zip")
    args = ap.parse_args()
    data = download(args.zip)
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xlsx_name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(
            io.BytesIO(z.read(xlsx_name)), sheet_name=None, dtype={"Invoice": str, "StockCode": str}
        )
    first, second = sheets.values()
    overlap = second["Invoice"].isin(set(first["Invoice"]))
    df = pd.concat([first, second[~overlap]], ignore_index=True)
    df.to_csv(CSV, index=False)

    write_json(
        REPO_ROOT / "data/manifest.json",
        {
            "source": "UCI Machine Learning Repository, Online Retail II (Chen, 2012)",
            "doi": "10.24432/C5CG6D",
            "license": "CC BY 4.0",
            "url": URL,
            "zip_sha256": SHA256,
            "sheets": {name: len(s) for name, s in sheets.items()},
            "dropped_overlap_rows": int(overlap.sum()),
            "rows": len(df),
            "columns": list(df.columns),
            "csv_sha256": sha256(CSV.read_bytes()),  # same as Data-Analyst-Agent's data/manifest.json
            "date_min": str(df["InvoiceDate"].min()),
            "date_max": str(df["InvoiceDate"].max()),
        },
    )
    print(f"wrote {CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
