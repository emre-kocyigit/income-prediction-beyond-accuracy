import os

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_col = config["target"]["column"]
    models_dir = config["paths"]["models"].rstrip("/")
    os.makedirs(models_dir, exist_ok=True)

    train = pd.read_csv(config["data"]["processed_train"])
    test = pd.read_csv(config["data"]["processed_test"])

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    continuous_features = [
        "age",
        "hours-per-week",
        "capital_gain_log",
        "capital_loss_log",
    ]
    binary_features = [
        "has_capital_gain",
        "has_capital_loss",
        "is_married",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "native_region",
        "age_group",
        "hours_category",
    ]

    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous_features),
            ("binary", "passthrough", binary_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", "passthrough", continuous_features),
            ("binary", "passthrough", binary_features),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_features,
            ),
        ]
    )

    lr_pipeline = Pipeline(
        [
            ("preprocessor", lr_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    try:
        import xgboost as xgb

        xgb_pipeline = Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    ),
                ),
            ]
        )
    except Exception:
        xgb_pipeline = None

    try:
        import lightgbm as lgb

        lgb_pipeline = Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except Exception:
        lgb_pipeline = None

    artifacts: dict[str, object] = {
        "logistic_regression": lr_pipeline,
        "random_forest": rf_pipeline,
    }
    if xgb_pipeline is not None:
        artifacts["xgboost"] = xgb_pipeline
    if lgb_pipeline is not None:
        artifacts["lightgbm"] = lgb_pipeline

    for name, model in artifacts.items():
        model.fit(X_train, y_train)
        out_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, out_path)
        print(f"✓ saved {name} → {out_path}")

    # small sanity check that loaded models run predict_proba
    for name in artifacts:
        model = joblib.load(os.path.join(models_dir, f"{name}.pkl"))
        proba = model.predict_proba(X_test)[:, 1]
        print(f"✓ loaded {name}: predict_proba ok (n={len(proba)})")


if __name__ == "__main__":
    main()

