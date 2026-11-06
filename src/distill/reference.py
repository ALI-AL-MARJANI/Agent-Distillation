"""Reference answers for the evaluation questions, computed with plain pandas (no LLM, no hand-typed number).

q01-q20 and h01-h10 are copied unchanged from Data-Analyst-Agent (scripts/build_truth.py at commit ad9ed18).
h11-h30 are new held-out questions, written on 2026-11-06 before any teacher trace or training run.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def revenue(df: pd.DataFrame) -> pd.Series:
    return df["Quantity"] * df["Price"]


def top_label(s: pd.Series, lowest: bool = False) -> str:
    """Label of the max (or min), refusing ties so the question has exactly one right answer."""
    s = s.sort_values(ascending=lowest)
    if len(s) > 1 and s.iloc[0] == s.iloc[1]:
        raise ValueError(f"tie for first place: {s.index[0]!r} and {s.index[1]!r}")
    return str(s.index[0])


def by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["InvoiceDate"].dt.year == year]


def by_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    return df[(df["InvoiceDate"].dt.year == year) & (df["InvoiceDate"].dt.month == month)]


def is_cancel(df: pd.DataFrame) -> pd.Series:
    return df["Invoice"].str.startswith("C")


EVAL_REFERENCE: dict[str, Callable[[pd.DataFrame], object]] = {
    # --- copied from Data-Analyst-Agent (dev q01-q20, held-out h01-h10) ---
    "q01": lambda df: len(df),
    "q02": lambda df: df["Invoice"].nunique(),
    "q03": lambda df: df["Country"].nunique(),
    "q04": lambda df: df["Price"].mean(),
    "q05": lambda df: top_label(
        df[df["Country"] != "United Kingdom"].groupby("Country")["Customer ID"].nunique()
    ),
    "q06": lambda df: top_label(revenue(df)[df["Country"] != "United Kingdom"].groupby(df["Country"]).sum()),
    "q07": lambda df: revenue(df)[df["Country"] == "Netherlands"].sum(),
    "q08": lambda df: top_label(df.groupby("Description")["Quantity"].sum()),
    "q09": lambda df: int((df["Quantity"] < 0).sum()),
    "q10": lambda df: df.loc[df["Invoice"].str.startswith("C"), "Invoice"].nunique(),
    "q11": lambda df: revenue(df)[df["Price"] > 100].sum(),
    "q12": lambda df: int(df["Customer ID"].isna().sum()),
    "q13": lambda df: top_label(revenue(df).groupby(df["InvoiceDate"].dt.strftime("%Y-%m")).sum()),
    "q14": lambda df: revenue(by_year(df, 2010)).sum(),
    "q15": lambda df: (
        (
            revenue(by_year(df[df["Country"] == "Germany"], 2011)).sum()
            / revenue(by_year(df[df["Country"] == "Germany"], 2010)).sum()
            - 1
        )
        * 100
    ),
    "q16": lambda df: top_label(df.drop_duplicates("Invoice")["InvoiceDate"].dt.day_name().value_counts()),
    "q17": lambda df: revenue(df)[df["Invoice"] == "489434"].sum(),
    "q18": lambda df: revenue(df).groupby(df["Customer ID"]).sum().mean(),  # groupby drops missing IDs
    "q19": lambda df: None,  # no cost column
    "q20": lambda df: None,  # no delivery dates
    "h01": lambda df: df["StockCode"].nunique(),
    "h02": lambda df: df["Quantity"].max(),
    "h03": lambda df: top_label(df.groupby("Country")["Price"].mean()),
    "h04": lambda df: df.loc[df["Country"] == "France", "Quantity"].sum(),
    "h05": lambda df: int((df["Price"] == 0).sum()),
    "h06": lambda df: revenue(df)[~df["Invoice"].str.startswith("C")].sum(),
    "h07": lambda df: df.loc[
        (df["InvoiceDate"].dt.year == 2011) & (df["InvoiceDate"].dt.month == 3), "Invoice"
    ].nunique(),
    "h08": lambda df: int(top_label(df["InvoiceDate"].dt.hour.value_counts())),
    "h09": lambda df: df.loc[df["Country"] == "Spain", "Customer ID"].nunique(),  # nunique skips missing IDs
    "h10": lambda df: None,  # no category column
    # --- new held-out questions (2026-11-06) ---
    "h11": lambda df: df["Price"].min(),
    "h12": lambda df: df["Customer ID"].nunique(),
    "h13": lambda df: top_label(df["StockCode"].value_counts()),
    "h14": lambda df: revenue(df)[df["Country"] == "Australia"].sum(),
    "h15": lambda df: float(top_label(revenue(df).groupby(df["Customer ID"]).sum())),
    "h16": lambda df: int((df["Country"] == "Germany").sum()),
    "h17": lambda df: int((df["Quantity"] > 100).sum()),
    "h18": lambda df: df.loc[df["Description"].str.contains("HEART", na=False), "Quantity"].sum(),
    "h19": lambda df: df.loc[is_cancel(df), "Customer ID"].nunique(),
    "h20": lambda df: df.loc[df["Quantity"] == 1, "Price"].mean(),
    "h21": lambda df: revenue(by_month(df, 2010, 12)).sum(),
    "h22": lambda df: int(top_label(df.groupby(df["InvoiceDate"].dt.year)["Invoice"].nunique())),
    "h23": lambda df: df.loc[df["InvoiceDate"].dt.day_name() == "Sunday", "Invoice"].nunique(),
    "h24": lambda df: top_label(
        by_year(df, 2010).groupby(df["InvoiceDate"].dt.strftime("%Y-%m"))["Quantity"].sum(), lowest=True
    ),
    "h25": lambda df: by_month(df, 2011, 11)["Customer ID"].nunique(),
    "h26": lambda df: len(df) / df["Invoice"].nunique(),
    "h27": lambda df: revenue(df).groupby(df["Invoice"]).sum().median(),
    "h28": lambda df: df["Customer ID"].isna().mean() * 100,
    "h29": lambda df: None,  # no age column
    "h30": lambda df: None,  # no payment column
}


def to_plain(value: object) -> object:
    """numpy scalar -> Python scalar, so it can be written to JSON."""
    return value.item() if hasattr(value, "item") else value
