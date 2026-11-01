# Personal Finance Tracker & Expense Predictor

A beginner-friendly, end-to-end Python project that turns bank CSV exports into categorized spending reports and a transparent next-month expense estimate. Everything runs locally—no bank API, credentials, or paid service required.

**Stack:** Python · Pandas · NumPy · Seaborn · Matplotlib · scikit-learn · pytest

## Features

- Read and validate CSV transactions using `pandas.read_csv()`.
- Categorize descriptions with editable, case-insensitive keyword rules.
- Separate income from expenses and account for explicitly labeled refunds.
- Summarize monthly net expenses by category.
- Create a stacked monthly bar chart, latest-month pie chart, and forecast chart.
- Fit a basic linear regression to monthly expense totals.
- Compare against a simple baseline on a chronological holdout when enough history exists.
- Allocate the estimated total to categories using recent spending shares.
- Export CSV summaries, JSON predictions, and PNG charts.
- Run automated tests and GitHub Actions CI.

## Preview

All example results use fictional data, not real financial records.

![Monthly spending](examples/sample-report/monthly_spending.png)
![Expense forecast](examples/sample-report/expense_forecast.png)
![Category share](examples/sample-report/category_share.png)

## 1. Install

Use Python 3.10 or newer. Run these commands from the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For runtime-only installation, use `python -m pip install -e .` instead.
Dependency ranges are declared in `pyproject.toml`; they are not an exact lockfile.

## 2. Run the demo

```bash
finance-tracker --input data/sample/bank_statement.csv
```

Equivalent module command:

```bash
python -m finance_tracker.cli --input data/sample/bank_statement.csv
```

The included sample contains **369 synthetic transactions across 18 complete months**, January 2024 through June 2025. Its next forecast month is **July 2025**, not the current calendar month: forecasts always follow the latest input month.

Example results, rounded:

- Next-month estimated net expenses: **USD 2,181.37**.
- Last-three-month holdout MAE: **USD 23.20**.
- The naive holdout baseline performs better on this sample; the application reports this explicitly rather than claiming regression is superior.

The command writes these outputs to `reports/`:

| Output | Contents |
|---|---|
| `categorized_transactions.csv` | Retained expenses/refunds, normalized amounts, month, category |
| `monthly_spending.csv` | Category columns indexed by month |
| `suggested_budget.csv` | Category shares and allocated next-month amounts |
| `forecast.json` | Prediction, trend, history range, warnings, holdout metrics |
| `monthly_spending.png` | Stacked monthly net spending |
| `category_share.png` | Latest-month positive net category spending |
| `expense_forecast.png` | Observed spending, fitted trend, next-month estimate |

Existing output files with these names are overwritten. Committed example outputs live separately in `examples/sample-report/`.

## 3. Use your own statement

Export transactions from your bank and rename columns to the schema below. Combine multiple monthly statements into one CSV with a single header. **One monthly statement alone cannot support a meaningful trend:** this demo requires at least **3 consecutive, complete months**, recommends more history, and evaluates a holdout only with at least 8 months.

```csv
date,description,amount,type
2025-01-01,Monthly salary,4200,income
2025-01-02,Rent landlord,-1250,expense
2025-01-05,Grocery supermarket,-85.50,expense
2025-01-06,Uber ride,-18.75,expense
2025-01-10,Supermarket return,12.00,refund
```

### Input contract

| Column | Required? | Meaning |
|---|---|---|
| `date` | Yes | ISO date `YYYY-MM-DD` |
| `description` | Yes | Nonblank merchant or transaction description |
| `amount` | Yes | Finite numeric value; no currency symbols or thousands separators |
| `type` | No | If present, every row must be `expense`, `refund`, or `income` |

**Choose your money convention carefully:**

1. **With `type`:** type overrides sign inference. Expenses become positive spending, refunds become negative spending, and income is removed. Magnitudes use the absolute input amount.
2. **Without `type`, default `--amount-mode signed`:** negative amounts are expenses; positive amounts are treated as income and removed. A positive refund cannot be distinguished from salary automatically—add `type` to preserve refunds.
3. **Without `type`, `--amount-mode expense-positive`:** positive amounts are expenses and negative amounts are refunds. Use this only for an expense-only export; remove income beforehand or add `type`.

Zero-value rows are removed. Dates, amounts, descriptions, types, and rule formats are validated; bad rows fail loudly instead of silently disappearing. Duplicate transactions are **not** automatically removed because identical legitimate purchases can occur. Reconcile overlaps before combining exports.

Put real statements in the ignored `data/private/` directory:

```bash
finance-tracker --input data/private/my_statement.csv --output reports/personal
```

For a positive-expense export:

```bash
finance-tracker --input data/private/expenses.csv --amount-mode expense-positive
```

Choose custom categorization rules and a display currency label:

```bash
finance-tracker --input data/private/my_statement.csv \
  --rules config/categories.json --currency EUR --output reports/personal
```

`--currency` changes labels only. It does not convert money; use a single currency per input.

## 4. Customize categories

Edit `config/categories.json`:

```json
{
  "Rent": ["rent", "landlord"],
  "Food": ["grocery", "restaurant", "cafe"],
  "Transport": ["uber", "lyft", "metro"],
  "Entertainment": ["netflix", "spotify", "cinema"]
}
```

Rules use **substring matching**, not machine learning. Categories are tested in JSON insertion order; the **first matching category wins**. Unmatched descriptions become `Other`. Review that category and improve rules over time. For example, put a specific `uber eats` rule under Food before a general Transport rule for `uber`. Short keywords can cause false matches.

## How the prediction works

1. Sum normalized expenses minus refunds for each calendar month.
2. Represent observed months as `0, 1, 2, ..., n-1`.
3. Fit scikit-learn `LinearRegression` to `monthly_total = intercept + slope × month_index`.
4. Predict at index `n`, corresponding to the next month.
5. Clip a negative predicted total to zero.

With 8 or more months, the last 3 months are held out. A separate model is trained only on earlier months; its MAE is compared with a constant prediction equal to the last training month's spending. This is a **fixed-origin three-month holdout**, not a rolling backtest. The final next-month model is then fitted on all provided months. No future observations enter holdout training.

MAE is mean absolute error in the input currency; smaller is better. It is not a confidence interval or a guarantee of future performance.

### Suggested category budget

The next-month total is allocated in proportion to each category's **trailing-three-month net spending**, clipped at zero **after summing those months**. Negative-share categories get zero allocation. Rounding is corrected so allocations sum to the predicted total. If all shares are zero, the total is marked `Unallocated`.

These are historical spending allocations—not individual category predictions, affordability advice, savings targets, or an income-aware budget. Review fixed bills, goals, and upcoming purchases yourself.

### Assumptions and limitations

- Only complete months should be included. The program cannot infer statement completeness from transaction dates; remove an incomplete current month yourself.
- Missing calendar months trigger an error. A month with no transactions is not silently filled with zero. The CSV-only interface cannot distinguish a genuinely empty month from missing data.
- No seasonality, uncertainty intervals, recurring-bill modeling, or outlier removal.
- Refunds reduce spending in their posting month, not the original purchase month.
- Negative category values are shown in the stacked chart. The pie chart excludes nonpositive categories, so it is not a full reconciliation of net spending.
- A simple trend can be worse than a naive forecast, especially with short or irregular history.
- This project is educational and is not financial advice.

## Project structure

```text
finance-tracker/
├── .github/workflows/tests.yml
├── config/categories.json
├── data/sample/bank_statement.csv
├── examples/sample-report/           # Fictional demo results and charts
├── src/finance_tracker/
│   ├── __init__.py
│   ├── data.py                      # CSV validation and sign normalization
│   ├── categorize.py                # Rules and monthly aggregation
│   ├── forecast.py                  # Regression, holdout, allocation
│   ├── visualize.py                 # Noninteractive PNG charts
│   └── cli.py                       # End-to-end command
├── tests/test_tracker.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Testing

```bash
python -m pytest -q
ruff check .
```

Tests cover sign conventions, income exclusion, refunds, invalid inputs, keyword precedence, monthly aggregation, exact linear trends, missing/insufficient history, nonnegative predictions, allocation rounding, and the full CLI workflow. GitHub Actions runs tests, linting, and a sample execution on Python 3.10 and 3.12.

## Upload to GitHub

Create an **empty** repository in GitHub, then run from this folder (replace the remote with your actual repository URL):

```bash
git init
git add .
git status                  # Review carefully: no real bank data should appear
git commit -m "Add end-to-end personal finance tracker"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/personal-finance-tracker.git
git push -u origin main
```

Suggested description: “Local Python expense tracker with rule-based categorization, spending visualizations, and a transparent linear-regression forecast.”

### Privacy

- Only synthetic sample data and its reports belong in the public repository.
- `data/private/`, `reports/`, `.env`, and virtual environments are ignored.
- Ignore rules do not protect previously tracked files or files stored elsewhere. Review `git status` and staged diffs before every commit.
- Exported reports can contain sensitive descriptions. Do not share personal outputs; avoid opening untrusted CSV descriptions in spreadsheets that execute formulas.
- There are no application-level bank/API/network calls; dependency installation still requires package downloads unless available locally.

## Possible extensions

- Add an interactive Streamlit dashboard.
- Import multiple CSVs with bank-specific column mappings.
- Add recurring-expense detection and merchant overrides.
- Compare moving-average, seasonal-naive, and regression forecasts with rolling evaluation.
- Track income, savings goals, and affordability separately from expense prediction.

## License

MIT. See `LICENSE`.
