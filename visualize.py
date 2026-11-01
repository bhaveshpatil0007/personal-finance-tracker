"""Noninteractive charts suitable for local runs and GitHub Actions."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_charts(monthly: pd.DataFrame, forecast: dict, out: Path, currency: str) -> None:
    sns.set_theme(style="whitegrid", palette="Set2")
    ax = monthly.plot.bar(stacked=True, figsize=(12, 6), width=0.8)
    ax.set(title="Monthly net spending by category", xlabel="Month",
           ylabel=f"Net expenses ({currency})")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "monthly_spending.png", dpi=160)
    plt.close(ax.figure)

    last = monthly.iloc[-1]
    positive = last[last > 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    if positive.empty:
        ax.text(0.5, 0.5, "No positive category spending", ha="center")
        ax.axis("off")
    else:
        ax.pie(positive, labels=positive.index, autopct="%1.1f%%", startangle=90,
               colors=sns.color_palette("Set2", len(positive)))
    ax.set_title(f"{monthly.index[-1]}: positive net category spending\n"
                 "Nonpositive categories excluded from pie")
    fig.tight_layout()
    fig.savefig(out / "category_share.png", dpi=160)
    plt.close(fig)

    totals = monthly.sum(axis=1)
    x = np.arange(len(totals))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, totals.values, marker="o", label="Actual net expenses")
    trend = forecast["monthly_trend"]
    # Plot a trend consistent with the fitted mean, using reported rounded slope.
    fitted = totals.mean() + trend * (x - x.mean())
    ax.plot(x, fitted, linestyle="--", label="Fitted linear trend")
    ax.scatter([len(x)], [forecast["predicted_expenses"]], marker="*", s=180,
               color="#d35400", label="Next month (estimate)")
    ax.set_xticks(list(x) + [len(x)], list(totals.index) + [forecast["forecast_month"]],
                  rotation=45, ha="right")
    ax.set(title="Total expenses and next-month forecast", ylabel=f"Net expenses ({currency})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "expense_forecast.png", dpi=160)
    plt.close(fig)
