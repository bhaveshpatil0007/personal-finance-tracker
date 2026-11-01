"""CSV validation and explicit money-sign normalization."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_transactions(path: str | Path, amount_mode: str = "signed") -> pd.DataFrame:
    """Return expenses/refunds with positive spending and negative refund amounts.

    signed: negative bank amounts are expenses, positive amounts are income.
    expense-positive: positive amounts are expenses, negative amounts are refunds.
    Optional type (expense/refund/income) overrides sign-based inference; amounts
    for explicitly typed rows use their absolute magnitude.
    """
    if amount_mode not in {"signed", "expense-positive"}:
        raise ValueError("amount_mode must be signed or expense-positive")
    df = pd.read_csv(path, dtype={"date": "string", "description": "string"})
    required = {"date", "description", "amount"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if df.empty:
        raise ValueError("The input CSV is empty")
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid dates: use YYYY-MM-DD for every row")
    amounts = pd.to_numeric(df["amount"], errors="coerce")
    if amounts.isna().any() or not np.isfinite(amounts).all():
        raise ValueError("Amounts must be finite numbers without currency symbols")
    descriptions = df["description"].str.strip()
    if descriptions.isna().any() or descriptions.eq("").any():
        raise ValueError("Descriptions must not be blank")
    if "type" in df.columns:
        kinds = df["type"].astype("string").str.strip().str.lower()
        if not kinds.isin(["expense", "refund", "income"]).all():
            raise ValueError("Every type must be expense, refund, or income")
        spending = amounts.abs().where(kinds.ne("refund"), -amounts.abs())
        keep = kinds.ne("income")
    elif amount_mode == "signed":
        keep = amounts.lt(0)
        spending = -amounts
    else:
        keep = amounts.ne(0)
        spending = amounts
    result = pd.DataFrame({"date": dates, "description": descriptions, "amount": spending})
    result = result.loc[keep & spending.ne(0)].copy()
    if result.empty:
        raise ValueError("No expenses or refunds remain after filtering")
    result["month"] = result["date"].dt.to_period("M").astype(str)
    return result.sort_values("date", kind="stable").reset_index(drop=True)
