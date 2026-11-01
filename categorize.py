"""Deterministic, first-match-wins substring classification."""
import json
from pathlib import Path

import pandas as pd


def load_rules(path: str | Path) -> dict[str, list[str]]:
    rules = json.loads(Path(path).read_text())
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Rules must be a nonempty JSON object")
    for category, keywords in rules.items():
        if (not isinstance(category, str) or not category.strip()
                or not isinstance(keywords, list) or not keywords
                or any(not isinstance(k, str) or not k.strip() for k in keywords)):
            raise ValueError("Each category needs a nonempty list of nonblank keyword strings")
    return {category: [word.strip().casefold() for word in words]
            for category, words in rules.items()}


def categorize(df: pd.DataFrame, rules: dict[str, list[str]]) -> pd.DataFrame:
    def match(description: str) -> str:
        text = description.casefold()
        return next((cat for cat, words in rules.items()
                     if any(word in text for word in words)), "Other")
    result = df.copy()
    result["category"] = result["description"].map(match)
    return result


def monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["month", "category"])["amount"].sum()
            .unstack(fill_value=0).sort_index().round(2))
