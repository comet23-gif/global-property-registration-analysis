"""Reproduce the notebook's held-out ensemble model comparison.

Usage: python src/modeling.py path/to/worlddata.csv
"""
import argparse

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

FEATURES = [
    "GDP per capita (current US$)",
    "Time required to start a business (days)",
    "Procedures to register property (number)",
]
TARGET = "Time required to register property (days)"


def prepare_data(path):
    data = pd.read_csv(path)
    selected = data[data["Year"].between(2004, 2014) & data["IndicatorName"].isin(FEATURES + [TARGET])]
    wide = selected.pivot_table(index=["CountryName", "CountryCode", "Year"],
                                columns="IndicatorName", values="Value").reset_index()
    return wide.dropna(subset=FEATURES + [TARGET])


def compare_models(data):
    train_x, test_x, train_y, test_y = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    models = {
        "Bagging": RandomForestRegressor(max_features=3, random_state=23),
        "Random Forest": RandomForestRegressor(max_features=2, random_state=23),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=23),
    }
    rows = []
    for name, model in models.items():
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        rows.append({"Model": name, "RMSE (days)": mean_squared_error(test_y, predictions) ** 0.5,
                     "MAE (days)": mean_absolute_error(test_y, predictions)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to worlddata.csv")
    args = parser.parse_args()
    data = prepare_data(args.data)
    print(f"Complete country-year observations: {len(data)}")
    print(compare_models(data).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
