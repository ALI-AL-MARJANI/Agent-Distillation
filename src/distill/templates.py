"""Training questions: templates x parameters on the same dataset, each with a pandas reference answer.

The templates teach the same skills as the benchmark (filters, group-bys, dates, refusing unanswerable
questions), but with other wording and other parameters. Where a template could rebuild an evaluation question,
the evaluation's parameter is excluded (e.g. revenue for the Netherlands, q07). tests/test_split.py checks that no
training question is a copy or a near-copy of an evaluation question.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from distill.reference import by_month, by_year, is_cancel, revenue, to_plain, top_label
from distill.runinfo import DATA

COUNTRIES = [
    "EIRE", "Germany", "France", "Netherlands", "Spain", "Switzerland", "Belgium", "Portugal", "Australia",
    "Channel Islands", "Italy", "Sweden", "Norway", "Cyprus", "Finland", "Austria", "Denmark", "Greece", "Japan",
    "USA", "Poland", "Israel", "Singapore", "Canada", "Iceland", "Malta", "Hong Kong", "Lithuania", "Brazil",
    "United Arab Emirates",
]  # fmt: skip
MONTHS = [(y, m) for y in (2010, 2011) for m in range(1, 13) if (y, m) != (2011, 12)]  # 2011-12 is partial
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]  # Sunday is h23
WORDS = [
    "BAG",
    "CANDLE",
    "MUG",
    "LIGHT",
    "BOX",
    "CHRISTMAS",
    "VINTAGE",
    "CARD",
    "GLASS",
    "CAKE",
    "PINK",
    "STAR",
]
MISSING_FIELDS = [
    "customer's gender", "reason for each return", "discount code used", "warehouse stock level",
    "supplier", "shipping fee", "customer star rating", "salesperson", "marketing spend", "number of website visits",
]  # fmt: skip


def month_name(y: int, m: int) -> str:
    return pd.Timestamp(year=y, month=m, day=1).strftime("%B %Y")


def country(df: pd.DataFrame, c: str) -> pd.DataFrame:
    return df[df["Country"] == c]


def countries(*exclude: str) -> list[dict]:
    return [{"c": c} for c in COUNTRIES if c not in exclude]


def months(*exclude: tuple[int, int]) -> list[dict]:
    return [{"y": y, "m": m, "month": month_name(y, m)} for y, m in MONTHS if (y, m) not in exclude]


def stock_codes(df: pd.DataFrame) -> list[dict]:
    return [{"s": s} for s in df["StockCode"].value_counts().index[:40]]


def invoices(df: pd.DataFrame) -> list[dict]:
    """Every 2,000th normal invoice: deterministic, spread over the two years. 489434 is q17."""
    normal = df.loc[~is_cancel(df), "Invoice"].drop_duplicates()
    return [{"inv": i} for i in normal.iloc[::2000] if i != "489434"]


@dataclass(frozen=True)
class Template:
    name: str
    category: str
    kind: str  # number | string | unanswerable
    phrasings: tuple[str, ...]
    params: Callable[[pd.DataFrame], list[dict]]
    answer: Callable[..., object]  # answer(df, **params)


TEMPLATES = [
    # simple
    Template("rows_country", "simple", "number",
             ("How many lines in the data come from customers in {c}?",
              "Count the transaction rows where Country is {c}."),
             lambda df: countries("Germany"),  # h16
             lambda df, c: len(country(df, c))),
    Template("products_country", "simple", "number",
             ("How many different StockCode values were sold to {c}?",
              "Number of distinct products (StockCode) bought by customers located in {c}?"),
             lambda df: countries(),
             lambda df, c: country(df, c)["StockCode"].nunique()),
    Template("quantity_month", "simple", "number",
             ("What is the sum of the Quantity column for {month}?",
              "Add up Quantity over all rows dated in {month}. What do you get?"),
             lambda df: months(),
             lambda df, y, m, month: by_month(df, y, m)["Quantity"].sum()),
    # groupby
    Template("revenue_country", "groupby", "number",
             ("Compute the sales amount, Quantity multiplied by Price, summed over all rows for {c}.",
              "Sum Quantity * Price for Country == '{c}'. What is the result?"),
             lambda df: countries("Netherlands", "Australia"),  # q07, h14
             lambda df, c: revenue(country(df, c)).sum()),
    Template("customers_country", "groupby", "number",
             ("Count the unique Customer ID values among rows from {c}.",
              "How many individual customer IDs placed orders from {c}?"),
             lambda df: countries("Spain"),  # h09
             lambda df, c: country(df, c)["Customer ID"].nunique()),
    Template("top_product_country", "groupby", "string",
             ("In {c}, which Description has the largest summed Quantity?",
              "Which product Description sold the most units (sum of Quantity) to {c}?"),
             lambda df: countries(),
             lambda df, c: top_label(country(df, c).groupby("Description")["Quantity"].sum())),
    Template("avg_price_country", "groupby", "number",
             ("What is the mean unit Price over the rows where Country is {c}?",
              "Average of Price for {c} rows?"),
             lambda df: countries(),
             lambda df, c: country(df, c)["Price"].mean()),
    Template("top_country_month", "groupby", "string",
             ("During {month}, which country other than the United Kingdom had the most rows?",
              "Excluding United Kingdom, name the Country with the highest row count in {month}."),
             lambda df: months(),
             lambda df, y, m, month: top_label(
                 by_month(df, y, m).query("Country != 'United Kingdom'")["Country"].value_counts())),
    # filter
    Template("rows_quantity_above", "filter", "number",
             ("Count rows whose Quantity exceeds {x}.",
              "In how many rows is Quantity strictly larger than {x}?"),
             lambda df: [{"x": x} for x in (5, 10, 12, 24, 36, 48, 50, 72, 144, 200, 500, 1000)],  # 100 is h17
             lambda df, x: int((df["Quantity"] > x).sum())),
    Template("revenue_price_above", "filter", "number",
             ("Sum of Quantity * Price over rows with a unit Price higher than {x}?",
              "Restrict to rows where Price > {x} and add up Quantity times Price."),
             lambda df: [{"x": x} for x in (2, 5, 8, 10, 15, 20, 30, 50, 75, 200, 500)],  # 100 is q11
             lambda df, x: revenue(df[df["Price"] > x]).sum()),
    Template("cancel_invoices_country", "filter", "number",
             ("How many distinct cancellation invoices (Invoice starting with 'C') come from {c}?",
              "For {c}, count the unique invoice numbers that begin with C."),
             lambda df: countries(),
             lambda df, c: country(df, c).loc[is_cancel(country(df, c)), "Invoice"].nunique()),
    Template("quantity_stockcode", "filter", "number",
             ("What is the total Quantity recorded for StockCode {s}?",
              "Sum the Quantity of all rows with StockCode '{s}'."),
             stock_codes,
             lambda df, s: df.loc[df["StockCode"] == s, "Quantity"].sum()),
    Template("rows_description_word", "filter", "number",
             ("How many rows have a Description containing '{w}'?",
              "Count the rows where the text {w} appears in Description."),
             lambda df: [{"w": w} for w in WORDS],
             lambda df, w: int(df["Description"].str.contains(w, na=False, regex=False).sum())),
    # dates
    Template("revenue_month", "dates", "number",
             ("Total of Quantity * Price for invoices dated in {month}?",
              "How much money (Quantity times Price, summed) was billed in {month}?"),
             lambda df: months((2010, 12)),  # h21
             lambda df, y, m, month: revenue(by_month(df, y, m)).sum()),
    Template("invoices_month", "dates", "number",
             ("Number of unique invoices issued in {month}?",
              "Count the distinct Invoice values with an InvoiceDate in {month}."),
             lambda df: months((2011, 3)),  # h07
             lambda df, y, m, month: by_month(df, y, m)["Invoice"].nunique()),
    Template("customers_month", "dates", "number",
             ("How many unique Customer ID values appear in {month}?",
              "Count distinct customers with at least one row dated {month}."),
             lambda df: months((2011, 11)),  # h25
             lambda df, y, m, month: by_month(df, y, m)["Customer ID"].nunique()),
    Template("best_month_country", "dates", "string",
             ("For {c}, in which month was the sum of Quantity * Price the highest? Reply as YYYY-MM.",
              "Find the YYYY-MM month with the largest Quantity times Price total for {c}."),
             lambda df: countries(),
             lambda df, c: top_label(
                 revenue(country(df, c)).groupby(country(df, c)["InvoiceDate"].dt.strftime("%Y-%m")).sum())),
    Template("revenue_change_country", "dates", "number",
             ("Percent change of the Quantity * Price total for {c}, 2011 compared with 2010 (e.g. -3.2 for -3.2 %)?",
              "By what percentage did {c}'s summed Quantity times Price move between 2010 and 2011? Example: 7.5 means +7.5 %."),
             lambda df: countries("Germany"),  # q15
             lambda df, c: (revenue(by_year(country(df, c), 2011)).sum()
                            / revenue(by_year(country(df, c), 2010)).sum() - 1) * 100),
    Template("invoices_weekday_year", "dates", "number",
             ("How many unique invoices were created on a {d} in {y}?",
              "In {y}, count the distinct invoices whose InvoiceDate falls on a {d}."),
             lambda df: [{"d": d, "y": y} for d in WEEKDAYS for y in (2010, 2011)],
             lambda df, d, y: by_year(df, y).loc[lambda x: x["InvoiceDate"].dt.day_name() == d, "Invoice"].nunique()),
    # tricky: Invoice is text, missing Customer ID, per-invoice aggregation, questions the data can't answer
    Template("invoice_value", "tricky", "number",
             ("What is the sum of Quantity * Price for invoice number {inv}?",
              "Invoice {inv}: how much is it worth in total (Quantity times Price over its lines)?"),
             invoices,
             lambda df, inv: revenue(df[df["Invoice"] == inv]).sum()),
    Template("avg_invoice_value_country", "tricky", "number",
             ("Average order value in {c}: sum Quantity * Price per invoice, then take the mean over invoices.",
              "For {c}, what is the mean of the per-invoice totals of Quantity times Price?"),
             lambda df: countries(),
             lambda df, c: revenue(country(df, c)).groupby(country(df, c)["Invoice"]).sum().mean()),
    Template("missing_customer_country", "tricky", "number",
             ("How many rows from {c} have an empty Customer ID?",
              "Count the {c} rows where Customer ID is missing."),
             lambda df: countries(),
             lambda df, c: int(country(df, c)["Customer ID"].isna().sum())),
    Template("missing_field_country", "tricky", "unanswerable",
             ("What is the most common {f} for orders from {c}?",
              "Using the data, report the {f} for customers in {c}."),
             lambda df: [{"f": f, "c": c} for f in MISSING_FIELDS for c in COUNTRIES[:6]],
             lambda df, f, c: None),
    Template("missing_field_product", "tricky", "unanswerable",
             ("What is the average {f} for StockCode {s}?",
              "Which {f} is linked to product {s}?"),
             lambda df: [{"f": f, **s} for f in MISSING_FIELDS for s in stock_codes(df)[:6]],
             lambda df, f, s: None),
]  # fmt: skip


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(DATA / "raw/online_retail_ii.csv", parse_dates=["InvoiceDate"])
    assert df["Invoice"].dtype != "int64", "Invoice should be read as text"
    return df


def valid_value(value: object, kind: str) -> bool:
    if kind == "unanswerable":
        return value is None
    if isinstance(value, float):
        return math.isfinite(value)
    return value is not None


def generate(df: pd.DataFrame, n: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    """Round-robin over the templates so every skill is represented; ties and NaN answers are skipped.
    Returns the questions and how many candidates were skipped per template."""
    rng = random.Random(seed)
    pools = []
    for t in TEMPLATES:
        combos = t.params(df)
        rng.shuffle(combos)
        pools.append((t, combos))
    out: list[dict] = []
    seen: set[str] = set()
    skipped: dict[str, int] = {}
    while len(out) < n and any(combos for _, combos in pools):
        for t, combos in pools:
            if not combos or len(out) >= n:
                continue
            p = combos.pop()
            text = rng.choice(t.phrasings).format(**p)
            try:
                value = to_plain(t.answer(df, **p))
            except ValueError:  # tie for first place: no single right answer
                value = float("nan")
            if text in seen or not valid_value(value, t.kind):
                skipped[t.name] = skipped.get(t.name, 0) + 1
                continue
            seen.add(text)
            out.append({"template": t.name, "category": t.category, "kind": t.kind, "question": text,
                        "params": p, "truth": {"kind": t.kind, "value": value}})  # fmt: skip
    return out, skipped


def assign_splits(questions: list[dict], n_valid: int, seed: int) -> list[dict]:
    """Give ids and a train/valid split at the question level (all steps of a trace stay in one split)."""
    valid = set(random.Random(seed).sample(range(len(questions)), n_valid))
    return [
        {"id": f"t{i + 1:03d}", "split": "valid" if i in valid else "train", **q}
        for i, q in enumerate(questions)
    ]
