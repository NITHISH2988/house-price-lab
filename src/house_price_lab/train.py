"""Compare simple regression baselines on the reproducible synthetic dataset."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import make_dataset

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "age_years"]
CATEGORICAL_FEATURES = ["location", "property_type"]


def make_pipeline(model) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def _feature_magnitudes(fitted_pipeline: Pipeline) -> list[dict[str, float | str]]:
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = fitted_pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        method = "random_forest_impurity_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).reshape(-1))
        method = "absolute_standardized_coefficient"
    else:
        return []
    order = np.argsort(values)[::-1][:10]
    return [
        {
            "feature": str(names[index]),
            "magnitude": float(values[index]),
            "method": method,
        }
        for index in order
    ]


def train(
    output_dir: Path = Path("artifacts"), *, rows: int = 1200, seed: int = 42
) -> dict:
    if rows < 50:
        raise ValueError("rows must be at least 50 for the five-fold comparison")
    data = make_dataset(n=rows, seed=seed)
    features = data.drop(columns="price")
    target = data["price"]
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=seed
    )
    candidates = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge_alpha_10": Ridge(alpha=10),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    cv_mae = {
        name: float(
            -cross_val_score(
                make_pipeline(estimator),
                train_x,
                train_y,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean()
        )
        for name, estimator in candidates.items()
    }
    selected_name = min(cv_mae, key=cv_mae.get)
    selected_model = make_pipeline(candidates[selected_name]).fit(train_x, train_y)
    predictions = selected_model.predict(test_x)
    residuals = test_y.to_numpy() - predictions
    report = {
        "dataset": "generated synthetic demonstration data",
        "synthetic": True,
        "target_unit": "synthetic dollars",
        "rows": len(data),
        "seed": seed,
        "split": {
            "strategy": "80/20 holdout plus 5-fold CV within training",
            "random_state": seed,
        },
        "selected_model": selected_name,
        "cv_mae": cv_mae,
        "test_metrics": {
            "mae": float(mean_absolute_error(test_y, predictions)),
            "rmse": float(mean_squared_error(test_y, predictions) ** 0.5),
            "r2": float(r2_score(test_y, predictions)),
        },
        "error_analysis": {
            "mean_signed_error": float(np.mean(residuals)),
            "median_absolute_error": float(np.median(np.abs(residuals))),
            "p90_absolute_error": float(np.quantile(np.abs(residuals), 0.9)),
        },
        "feature_magnitudes": _feature_magnitudes(selected_model),
    }
    output_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(selected_model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.output, rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
