import json

import pandas as pd
import pytest

from finance_tracker.categorize import categorize, load_rules, monthly_by_category
from finance_tracker.cli import main
from finance_tracker.data import load_transactions
from finance_tracker.forecast import allocate_budget, forecast_next_month


def write_csv(tmp_path, text):
    path = tmp_path / "bank.csv"
    path.write_text(text)
    return path


def test_signed_filters_income(tmp_path):
    path = write_csv(tmp_path, "date,description,amount\n2025-01-01,Rent,-1000\n"
                     "2025-01-02,Salary,3000\n")
    df = load_transactions(path)
    assert df["amount"].tolist() == [1000]


def test_typed_refunds_and_income(tmp_path):
    path = write_csv(tmp_path, "date,description,amount,type\n2025-01-01,Shop,-100,expense\n"
                     "2025-01-02,Shop,20,refund\n2025-01-03,Salary,500,income\n")
    assert load_transactions(path)["amount"].tolist() == [100, -20]


def test_positive_mode(tmp_path):
    path = write_csv(tmp_path, "date,description,amount\n2025-01-01,Shop,100\n"
                     "2025-01-02,Refund,-10\n")
    assert load_transactions(path, "expense-positive")["amount"].sum() == 90


@pytest.mark.parametrize("row", ["2025-02-30,Rent,-10", "2025-01-01,Rent,abc",
                                  "2025-01-01,,10", "2025-01-01,Rent,inf"])
def test_invalid_rows(tmp_path, row):
    with pytest.raises(ValueError):
        load_transactions(write_csv(tmp_path, "date,description,amount\n" + row + "\n"))


def test_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing required"):
        load_transactions(write_csv(tmp_path, "date,amount\n2025-01-01,10\n"))


def test_invalid_type(tmp_path):
    with pytest.raises(ValueError, match="Every type"):
        load_transactions(write_csv(tmp_path, "date,description,amount,type\n"
                                    "2025-01-01,Shop,10,other\n"))


def test_rules_and_first_match(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"Transport": ["UBER"], "Food": ["eats"]}))
    df = pd.DataFrame({"description": ["Uber Eats", "unknown"]})
    result = categorize(df, load_rules(path))
    assert result["category"].tolist() == ["Transport", "Other"]
    assert "category" not in df


def test_bad_rules(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"Food": [""]}')
    with pytest.raises(ValueError):
        load_rules(path)


def history(n=12):
    return pd.DataFrame({"Food": [100 + 10 * i for i in range(n)]},
                        index=pd.period_range("2024-01", periods=n, freq="M").astype(str))


def test_exact_linear_forecast_and_holdout():
    result = forecast_next_month(history())
    assert result["predicted_expenses"] == 220
    assert result["forecast_month"] == "2025-01"
    assert result["evaluation"]["linear_regression_mae"] == 0
    assert result["evaluation"]["constant_last_training_month_mae"] == 20


def test_insufficient_history():
    with pytest.raises(ValueError, match="at least 3"):
        forecast_next_month(history(2))


def test_missing_months():
    with pytest.raises(ValueError, match="Missing months"):
        forecast_next_month(history().drop("2024-05"))


def test_negative_prediction_clamped():
    monthly = pd.DataFrame({"Food": [300, 200, 100]},
                           index=["2024-01", "2024-02", "2024-03"])
    assert forecast_next_month(monthly)["predicted_expenses"] == 0


def test_budget_rounding():
    monthly = pd.DataFrame({"A": [1], "B": [1], "C": [1]})
    budget = allocate_budget(monthly, 100)
    assert budget["suggested_budget"].sum() == pytest.approx(100)
    assert budget["share"].sum() == pytest.approx(1)


def test_no_positive_budget_history():
    budget = allocate_budget(pd.DataFrame({"A": [-1]}), 100)
    assert budget.iloc[0]["category"] == "Unallocated"
    assert budget["suggested_budget"].sum() == 100


def test_monthly_aggregation():
    df = pd.DataFrame({"month": ["2024-01"] * 2, "category": ["Food"] * 2,
                       "amount": [100, -10]})
    assert monthly_by_category(df).iloc[0, 0] == 90


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["finance-tracker", "--input",
                        "data/sample/bank_statement.csv", "--output", str(tmp_path)])
    assert main() == 0
    for name in ["categorized_transactions.csv", "monthly_spending.csv", "suggested_budget.csv",
                 "forecast.json", "monthly_spending.png", "category_share.png",
                 "expense_forecast.png"]:
        assert (tmp_path / name).stat().st_size > 0
    report = json.loads((tmp_path / "forecast.json").read_text())
    budgets = pd.read_csv(tmp_path / "suggested_budget.csv")
    assert budgets["suggested_budget"].sum() == pytest.approx(report["predicted_expenses"])
