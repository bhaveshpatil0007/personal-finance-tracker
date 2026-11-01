"""Simple trend forecast, with a chronological holdout and naive baseline."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def forecast_next_month(monthly: pd.DataFrame) -> dict:
    totals = monthly.sum(axis=1).sort_index()
    periods = pd.PeriodIndex(totals.index, freq="M")
    if len(totals) < 3:
        raise ValueError("Forecast requires at least 3 complete, consecutive months")
    if list(periods) != list(pd.period_range(periods.min(), periods.max(), freq="M")):
        raise ValueError("Missing months: provide consecutive history; do not assume gaps are zero")
    y = totals.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Monthly totals must be finite")
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    raw = float(model.predict([[len(y)]])[0])
    warnings = [
        "This is an illustrative trend forecast, not financial advice or a guaranteed budget.",
        "Only include complete months; partial months bias the forecast downward.",
        "A linear trend does not model seasonality, inflation, or unexpected expenses.",
    ]
    if len(y) < 6:
        warnings.append("Fewer than 6 months: the estimate is especially fragile.")
    if (y < 0).any():
        warnings.append("Some monthly net expenses are negative because refunds exceed expenses.")
    evaluation = None
    if len(y) >= 8:
        # Last three months are never used to fit the holdout model.
        split = len(y) - 3
        held_out = LinearRegression().fit(x[:split], y[:split])
        pred = np.maximum(held_out.predict(x[split:]), 0)
        evaluation = {
            "method": "last_3_months_chronological_holdout",
            "test_months": totals.index[split:].tolist(),
            "linear_regression_mae": round(float(mean_absolute_error(y[split:], pred)), 2),
            "constant_last_training_month_mae": round(float(
                mean_absolute_error(y[split:], np.repeat(y[split - 1], 3))), 2),
        }
        if evaluation["linear_regression_mae"] > evaluation["constant_last_training_month_mae"]:
            warnings.append("Linear regression underperformed the naive holdout baseline.")
    else:
        warnings.append("Holdout evaluation requires at least 8 months; no accuracy claim is made.")
    return {
        "forecast_month": str(periods[-1] + 1),
        "history_start": str(periods[0]), "history_end": str(periods[-1]),
        "months_used": len(y), "model": "LinearRegression(month_index -> net_expenses)",
        "predicted_expenses": round(max(0.0, raw), 2),
        "raw_prediction": round(raw, 2),
        "monthly_trend": round(float(model.coef_[0]), 2),
        "evaluation": evaluation, "warnings": warnings,
    }


def allocate_budget(monthly: pd.DataFrame, total: float) -> pd.DataFrame:
    """Allocate the forecast using positive trailing-three-month net category spend.

    These are suggested shares, not separately fitted category forecasts.
    """
    history = monthly.tail(3).sum().clip(lower=0)
    shares = history / history.sum() if history.sum() > 0 else history * 0
    if shares.sum() == 0:
        return pd.DataFrame({"category": ["Unallocated"], "share": [1.0],
                             "suggested_budget": [total]})
    cents = (shares * round(total * 100)).round().astype(int)
    # Correct rounding so category budgets add up exactly to the forecast.
    cents.loc[shares.idxmax()] += round(total * 100) - int(cents.sum())
    return pd.DataFrame({"category": shares.index, "share": shares.values,
                         "suggested_budget": cents.values / 100})
