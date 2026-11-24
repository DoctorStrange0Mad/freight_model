"""
Drift-aware freight rate model
==============================
Why this exists (see DIAGNOSIS.md for the evidence):

1. ~1.4% of labels are randomly corrupted (x0.2-0.5 or x2-5, classifier AUC 0.51 = unpredictable).
   They carry ~99.5% of the squared error, so we train with an outlier-robust loss (Huber) and
   predict the clean rate (the median), which is optimal for MAE / MAPE / MedAPE and ~optimal for R2.
2. Calendar features (day-of-year) look great under random K-fold but are pure leakage: neighbouring
   days share a latent market factor. They are NOT used.
3. After removing calendar features, the dominant forward error is a slow upward drift in the market
   level that trees cannot extrapolate. We estimate it from week-blocked out-of-fold residuals
   (no leakage) and forecast it as: robust linear trend + persistence of the last 14 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

EARTH_R = 3958.8
T0 = pd.Timestamp("2025-01-01")

BASE_FEATURES = ["pickup", "delivery", "equipment", "distance", "weight_c", "mi", "quote_signal"]
GEO_FEATURES = BASE_FEATURES + ["hav", "circ", "lane_med", "d_vs_lane"]
CATS = ["pickup", "delivery", "equipment"]


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _lgb(seed: int, n_estimators: int = 400, learning_rate: float = 0.06) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber",         # robust to the ~1.4% corrupted labels, but efficient on clean rows (beats LAD)
        alpha=0.06,                # Huber threshold in $/mile (~3% of a typical rate per mile)
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


class DriftAwareFreightModel:
    def __init__(self, seeds=(0, 1, 2), n_estimators: int = 400, learning_rate: float = 0.06,
                 level_days: int = 14, drift_clip: float = 0.06, use_drift: bool = True) -> None:
        self.seeds = tuple(seeds)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.level_days = level_days
        self.drift_clip = drift_clip
        self.use_drift = use_drift

    # ------------------------------------------------------------------ features (label-free)
    def _prepare(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        w = d["weight"].astype(float).abs()                       # negative weights are sign errors
        if fit:
            self.w_med_ = w.groupby(d["equipment"]).median().to_dict()
            self.w_global_ = float(w.median())
        w = w.fillna(d["equipment"].map(self.w_med_)).fillna(self.w_global_)
        d["weight_c"] = w

        mi = d["market_index"].astype(float)
        mi = mi.fillna(mi.groupby(d["date"]).transform("mean"))   # day-level signal, so day mean is the right fill
        if fit:
            self.mi_global_ = float(mi.median())
        d["mi"] = mi.fillna(self.mi_global_)
        d["quote_signal"] = d["quote_signal"].astype(float)

        d["hav"] = _haversine(d["pickup_lat"], d["pickup_lon"], d["delivery_lat"], d["delivery_lon"])
        d["circ"] = d["distance"] / (d["hav"] + 1.0)
        if fit:
            self.lane_med_ = d.groupby(["pickup", "delivery"])["distance"].median().to_dict()
            self.cats_ = {c: sorted(d[c].astype(str).unique()) for c in CATS}
        lm = pd.Series(list(zip(d["pickup"], d["delivery"])), index=d.index).map(self.lane_med_)
        d["lane_med"] = lm.fillna(d["distance"])                  # unseen lane -> its own distance
        d["d_vs_lane"] = d["distance"] / d["lane_med"]
        for c in CATS:
            d[c] = pd.Categorical(d[c].astype(str), categories=self.cats_[c])
        return d

    def _rpm(self, models_by_feat, d: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(d[feats]) for feats, ms in models_by_feat for m in ms]
        return np.mean(preds, axis=0)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "DriftAwareFreightModel":
        d = self._prepare(df, fit=True)
        y = (d["posted_rate"] / d["distance"]).values
        # honest (week-blocked) out-of-fold residuals: used for drift AND to measure each month's label noise
        r = self._oof_residuals(d, y)
        w = self._month_weights(d, r)
        self.models_ = [
            (feats, [_lgb(s, self.n_estimators, self.learning_rate).fit(d[feats], y, sample_weight=w) for s in self.seeds])
            for feats in (BASE_FEATURES, GEO_FEATURES)
        ]
        self.drift_ = self._fit_drift(d, r) if self.use_drift else None
        return self

    def _oof_residuals(self, d: pd.DataFrame, y: np.ndarray) -> pd.Series:
        weeks = d["date"].dt.isocalendar().week.astype(int).values
        oof = np.zeros(len(d))
        for tr_i, te_i in GroupKFold(6).split(d, groups=weeks):
            ms = [_lgb(0, self.n_estimators, self.learning_rate).fit(d.iloc[tr_i][f], y[tr_i])
                  for f in (BASE_FEATURES, GEO_FEATURES)]
            oof[te_i] = np.mean([m.predict(d.iloc[te_i][f]) for m, f in zip(ms, (BASE_FEATURES, GEO_FEATURES))], axis=0)
        return pd.Series(np.log(y / oof), index=d.index)

    def _month_weights(self, d: pd.DataFrame, r: pd.Series) -> np.ndarray:
        """Inverse-variance weights per month (August labels are ~3x noisier -> weight ~0.15)."""
        m = d["date"].dt.month
        rr = r[r.abs() < 0.15]
        # month-level level shifts (drift) must not count as noise: centre each month on its median
        sig = rr.groupby(m[rr.index]).apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
        w = np.clip((sig.median() / sig) ** 2, 0.1, 1.0)
        self.month_weights_ = w.to_dict()
        return m.map(w).fillna(1.0).values

    def _fit_drift(self, d: pd.DataFrame, r: pd.Series) -> dict:
        keep = r.abs() < 0.15                                      # drop corrupted labels / noise spikes
        daily = r[keep].groupby(d.loc[keep, "date"]).median()      # robust day-level shock
        t = ((daily.index - T0).days).values.astype(float)
        slope, _ = np.polyfit(t, daily.values, 1)
        icpt = float(np.median(daily.values - slope * t))
        last = d["date"].max()
        recent = keep & (d["date"] > last - pd.Timedelta(days=self.level_days))
        t_recent = ((d.loc[recent, "date"] - T0).dt.days).values.astype(float)
        dev = float(r[recent].mean() - (icpt + slope * t_recent.mean()))
        return {"slope": float(slope), "intercept": icpt, "deviation": dev, "last_date": last,
                "daily_shock": daily}

    def drift_for(self, dates) -> np.ndarray:
        if not self.use_drift:
            return np.zeros(len(dates))
        t = ((pd.to_datetime(pd.Series(dates)) - T0).dt.days).values.astype(float)
        s = self.drift_["intercept"] + self.drift_["slope"] * t + self.drift_["deviation"]
        return np.clip(s, -self.drift_clip, self.drift_clip)

    # ------------------------------------------------------------------ predict
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare(df, fit=False)
        rpm = self._rpm(self.models_, d)
        rate = rpm * d["distance"].values * np.exp(self.drift_for(d["date"].values))
        return np.maximum(rate, 10.0)
