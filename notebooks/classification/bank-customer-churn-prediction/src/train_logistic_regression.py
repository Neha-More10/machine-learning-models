from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/bank_churn_sample.csv")
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run `python src/make_sample_data.py` first."
        )
    return pd.read_csv(path)


def build_model() -> Pipeline:
    numeric_features = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary",
    ]
    categorical_features = ["country", "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def show_coefficients(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = pd.Series(classifier.coef_[0], index=feature_names)

    print("\nTop positive churn drivers:")
    print(coefficients.sort_values(ascending=False).head(8).round(3).to_string())

    print("\nTop negative churn drivers:")
    print(coefficients.sort_values().head(8).round(3).to_string())


def main() -> None:
    data = load_data()
    x = data.drop(columns=[TARGET, *DROP_COLUMNS])
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    churn_probabilities = model.predict_proba(x_test)[:, 1]

    print(f"Rows: {len(data):,}")
    print(f"Churn rate: {y.mean():.1%}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, churn_probabilities):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["stayed", "churned"]))

    print("Confusion matrix:")
    matrix = confusion_matrix(y_test, predictions)
    print(
        pd.DataFrame(
            matrix,
            index=["actual_stayed", "actual_churned"],
            columns=["pred_stayed", "pred_churned"],
        )
    )

    show_coefficients(model)


if __name__ == "__main__":
    main()
