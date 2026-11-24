from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.drift_model import DriftAwareFreightModel


def run_full_pipeline(
    train_path="data/train_test.csv",
    val_path="data/validation.csv",
    dec_path="data/december_chart_inputs.csv",
    predictions_template_path="data/validation_predictions_template.csv",
    output_predictions_path="validation_predictions.csv",
    model_save_path="reports/models/ensemble_model.joblib",
) -> None:
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    dec = pd.read_csv(dec_path)
    template = pd.read_csv(predictions_template_path)

    model = DriftAwareFreightModel().fit(train)
    joblib.dump(model, model_save_path)
    print("drift params:", {k: v for k, v in model.drift_.items() if k != "daily_shock"})

    val_pred = np.round(model.predict(val), 2)
    out = template.copy()
    out["predicted_rate"] = out["load_id"].map(dict(zip(val["load_id"], val_pred)))
    assert len(out) == 12000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(output_predictions_path, index=False)

    # December fixed lane: coordinates from the city lookup, market signals from the validation days (features only)
    pick = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    deli = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    d = dec.drop(columns=["predicted_rate"], errors="ignore").copy()
    d[["pickup_lat", "pickup_lon"]] = pick.loc[d["pickup"]].values
    d[["delivery_lat", "delivery_lon"]] = deli.loc[d["delivery"]].values
    v = val.assign(date=pd.to_datetime(val["date"]))
    d["date"] = pd.to_datetime(d["date"])
    d["market_index"] = d["date"].map(v.groupby("date")["market_index"].mean())
    d["quote_signal"] = d["date"].map(v[v["equipment"] == "Dry Van"].groupby("date")["quote_signal"].mean())
    dec_out = dec.copy()
    dec_out["predicted_rate"] = np.round(model.predict(d), 2)
    assert dec_out["predicted_rate"].notna().all()
    dec_out.to_csv(dec_path, index=False)
    print(dec_out["predicted_rate"].describe())


if __name__ == "__main__":
    run_full_pipeline()
