import os
from typing import Literal

import lorem
from fastapi import FastAPI

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
app = FastAPI(title="Inference API", version=MODEL_VERSION)


@app.get("/health")
def health() -> dict:
    if os.environ.get("BROKEN_HEALTH") == "1":
        raise RuntimeError("Проверка готовности сломана")
    return {"status": "ok"}


if MODEL_VERSION == "v1.0":
    @app.get("/infer")
    def infer() -> dict:
        if os.environ.get("BROKEN_INFER") == "1":
            raise RuntimeError("Модель сломана")
        return {"model_version": MODEL_VERSION, "result": lorem.sentence()}

elif MODEL_VERSION == "v2.0":
    generators = {
        "sentence": lorem.sentence,
        "paragraph": lorem.paragraph,
        "text": lorem.text,
    }

    @app.get("/infer")
    def infer(size: Literal["sentence", "paragraph", "text"] = "sentence") -> dict:
        return {"model_version": MODEL_VERSION, "result": generators[size]()}

elif MODEL_VERSION == "v3.0":
    from titanic import add_predict_route

    add_predict_route(app)
else:
    raise ValueError(f"Unsupported model version: {MODEL_VERSION}")
