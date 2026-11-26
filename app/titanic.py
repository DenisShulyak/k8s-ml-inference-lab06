import argparse
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch"]


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["female", "male"]
    Age: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)


def encode(frame: pd.DataFrame, age_median: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["Age"] = pd.to_numeric(frame["Age"]).fillna(age_median)
    return pd.get_dummies(frame, columns=["Sex"], dtype=int)


def add_predict_route(app: FastAPI) -> None:
    with Path("model.pkl").open("rb") as stream:
        artifact = pickle.load(stream)

    @app.post("/predict")
    def predict(passenger: Passenger) -> dict:
        frame = pd.DataFrame([passenger.model_dump()], columns=FEATURES)
        encoded = encode(frame, artifact["age_median"])
        encoded = encoded.reindex(columns=artifact["columns"], fill_value=0)
        fare = float(artifact["model"].predict(encoded)[0])
        return {"model_version": "v3.0", "variant": "titanic-fare", "fare_estimate": round(fare, 2)}


def train(csv: str, output: Path) -> None:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(csv)
    canonical = {name.lower(): name for name in [*FEATURES, "Fare"]}
    data = data.rename(columns=lambda name: canonical.get(name.strip().lower(), name))
    data = data.dropna(subset=["Pclass", "Sex", "SibSp", "Parch", "Fare"])
    train_rows, test_rows = train_test_split(data, test_size=0.2, random_state=42)
    age_median = float(train_rows["Age"].median())
    x_train = encode(train_rows[FEATURES], age_median)
    x_test = encode(test_rows[FEATURES], age_median).reindex(columns=x_train.columns, fill_value=0)

    model = LinearRegression()
    model.fit(x_train, train_rows["Fare"])
    mae = mean_absolute_error(test_rows["Fare"], model.predict(x_test))
    print(f"train={len(train_rows)}, test={len(test_rows)}, MAE={mae:.4f}")
    with output.open("wb") as stream:
        pickle.dump({"model": model, "columns": list(x_train.columns), "age_median": age_median}, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", type=Path, default=Path("model.pkl"))
    args = parser.parse_args()
    train(args.csv, args.output)
