from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def make_bank_churn_sample(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = rng.choice(["France", "Germany", "Spain"], size=rows, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Female", "Male"], size=rows, p=[0.48, 0.52])
    age = np.clip(rng.normal(39, 10, rows).round(), 18, 80).astype(int)
    tenure = rng.integers(0, 11, size=rows)
    credit_score = np.clip(rng.normal(650, 95, rows).round(), 350, 850).astype(int)
    products_number = rng.choice([1, 2, 3, 4], size=rows, p=[0.48, 0.38, 0.11, 0.03])
    credit_card = rng.choice([0, 1], size=rows, p=[0.3, 0.7])
    active_member = rng.choice([0, 1], size=rows, p=[0.48, 0.52])
    estimated_salary = rng.uniform(15000, 200000, size=rows).round(2)

    has_balance = rng.choice([0, 1], size=rows, p=[0.35, 0.65])
    balance = has_balance * rng.normal(105000, 43000, rows)
    balance = np.clip(balance, 0, 250000).round(2)

    churn_score = (
        -1.45
        + 0.065 * (age - 40)
        - 0.005 * (credit_score - 650)
        + 0.000008 * (balance - 75000)
        + 0.42 * (products_number == 1)
        + 1.1 * (products_number >= 3)
        - 1.2 * active_member
        + 0.36 * (countries == "Germany")
        + 0.18 * (gender == "Female")
        - 0.035 * tenure
    )
    churn_probability = sigmoid(churn_score)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": np.arange(100000, 100000 + rows),
            "credit_score": credit_score,
            "country": countries,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": estimated_salary,
            "churn": churn,
        }
    )


def main() -> None:
    output_path = Path("data/bank_churn_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = make_bank_churn_sample()
    data.to_csv(output_path, index=False)
    churn_rate = data["churn"].mean()

    print(f"Wrote {len(data):,} rows to {output_path}")
    print(f"Churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()
