"""Command-line entry point."""
import argparse
import json
import sys
from pathlib import Path

from .categorize import categorize, load_rules, monthly_by_category
from .data import load_transactions
from .forecast import allocate_budget, forecast_next_month
from .visualize import create_charts


def main() -> int:
    parser = argparse.ArgumentParser(description="Categorize a bank CSV and forecast next month.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with date,description,amount")
    parser.add_argument("--rules", type=Path, default=Path("config/categories.json"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--amount-mode", choices=["signed", "expense-positive"], default="signed")
    parser.add_argument("--currency", default="USD", help="Display label only; no conversion")
    args = parser.parse_args()
    try:
        df = categorize(load_transactions(args.input, args.amount_mode), load_rules(args.rules))
        monthly = monthly_by_category(df)
        forecast = forecast_next_month(monthly)
        forecast["currency"] = args.currency
        budget = allocate_budget(monthly, forecast["predicted_expenses"])
        args.output.mkdir(parents=True, exist_ok=True)
        df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(
            args.output / "categorized_transactions.csv", index=False)
        monthly.to_csv(args.output / "monthly_spending.csv")
        budget.to_csv(args.output / "suggested_budget.csv", index=False)
        (args.output / "forecast.json").write_text(json.dumps(forecast, indent=2) + "\n")
        create_charts(monthly, forecast, args.output, args.currency)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Next month ({forecast['forecast_month']}): "
          f"{args.currency} {forecast['predicted_expenses']:,.2f}")
    if forecast["evaluation"]:
        print("Holdout MAE:", forecast["evaluation"]["linear_regression_mae"])
    print(f"Reports saved to {args.output}")
    for warning in forecast["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
