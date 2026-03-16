"""
train.py — Taxi Trip Duration (v3)
====================================
Entraîne un GradientBoostingRegressor avec features base + lag/rolling.
Sauvegarde le modèle dans model.pkl.

Usage:
    python train.py --data train.csv [--output model.pkl]
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distance orthodromique en km."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


BASE_FEATURES = [
    "vendor_id", "passenger_count", "distance_km",
    "hour", "dayofweek", "month",
    "is_weekend", "is_rush_hour", "store_flag", "dist_x_hour",
]

LAG_FEATURES = [
    "global_lag1", "global_lag2",
    "global_roll3_mean", "global_roll5_std", "global_roll10_mean",
    "vendor_expand_mean", "vendor_expand_std",
    "hour_slot_mean", "dow_mean", "vendor_speed_mean",
]

ALL_FEATURES = BASE_FEATURES + LAG_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique tout le feature engineering sur un DataFrame brut.
    Trie par pickup_datetime (indispensable pour lag/rolling sans fuite).
    """
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    # ── Base ──────────────────────────────────────────────────
    df["distance_km"]  = haversine(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["hour"]         = df["pickup_datetime"].dt.hour
    df["dayofweek"]    = df["pickup_datetime"].dt.dayofweek
    df["month"]        = df["pickup_datetime"].dt.month
    df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["store_flag"]   = (df["store_and_fwd_flag"] == "Y").astype(int)
    df["dist_x_hour"]  = df["distance_km"] * df["hour"]
    df["hour_slot"]    = df["hour"] // 3

    # ── Lag & Rolling (shift(1) = pas de fuite temporelle) ────
    gm = df["trip_duration"].mean() if "trip_duration" in df.columns else 800.0

    s = df["trip_duration"].shift(1) if "trip_duration" in df.columns \
        else pd.Series(gm, index=df.index)

    df["global_lag1"]        = s.fillna(gm)
    df["global_lag2"]        = (df["trip_duration"].shift(2).fillna(gm)
                                if "trip_duration" in df.columns else gm)
    df["global_roll3_mean"]  = s.rolling(3,  min_periods=1).mean().fillna(gm)
    df["global_roll5_std"]   = s.rolling(5,  min_periods=2).std().fillna(0)
    df["global_roll10_mean"] = s.rolling(10, min_periods=3).mean().fillna(gm)

    if "trip_duration" in df.columns:
        df["vendor_expand_mean"] = (
            df.groupby("vendor_id")["trip_duration"]
              .transform(lambda x: x.shift(1).expanding().mean()).fillna(gm))
        df["vendor_expand_std"] = (
            df.groupby("vendor_id")["trip_duration"]
              .transform(lambda x: x.shift(1).expanding().std()).fillna(0))
        df["hour_slot_mean"] = (
            df.groupby("hour_slot")["trip_duration"]
              .transform(lambda x: x.shift(1).expanding().mean()).fillna(gm))
        df["dow_mean"] = (
            df.groupby("dayofweek")["trip_duration"]
              .transform(lambda x: x.shift(1).expanding().mean()).fillna(gm))
        df["speed_proxy"] = df["distance_km"] / df["trip_duration"].replace(0, np.nan)
        df["vendor_speed_mean"] = (
            df.groupby("vendor_id")["speed_proxy"]
              .transform(lambda x: x.shift(1).expanding().mean())
              .fillna(df["speed_proxy"].mean()))
    else:
        for col in ["vendor_expand_mean", "vendor_expand_std",
                    "hour_slot_mean", "dow_mean"]:
            df[col] = gm if "std" not in col else 0
        df["vendor_speed_mean"] = df["distance_km"] / gm

    return df


def remove_outliers(df: pd.DataFrame,
                    min_s: int = 60, max_s: int = 7200) -> pd.DataFrame:
    before = len(df)
    df = df[(df["trip_duration"] >= min_s) & (df["trip_duration"] <= max_s)].copy()
    removed = before - len(df)
    print(f"[clean] {removed} outliers supprimés ({removed/before*100:.1f}%)  "
          f"→  {len(df):,} lignes")
    return df


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(data_path: str, output_path: str = "model.pkl") -> None:
    print(f"[train] Chargement : {data_path}")
    df = pd.read_csv(data_path, index_col=0)
    df = remove_outliers(df)
    df = build_features(df)

    X = df[ALL_FEATURES]
    y = df["trip_duration"]

    # Split chronologique strict
    split = int(len(df) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    print(f"[train] Train={len(X_tr):,}  Val={len(X_val):,}")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    print("[train] Entraînement …")
    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    mae   = mean_absolute_error(y_val, preds)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    r2    = r2_score(y_val, preds)

    print("\n── Métriques validation ──────────────────────────")
    print(f"  MAE  = {mae:>8.1f} s  (~{mae/60:.1f} min)")
    print(f"  RMSE = {rmse:>8.1f} s  (~{rmse/60:.1f} min)")
    print(f"  R²   = {r2:>8.4f}")
    print("──────────────────────────────────────────────────")

    artefact = {
        "model":         model,
        "features":      ALL_FEATURES,
        "base_features": BASE_FEATURES,
        "lag_features":  LAG_FEATURES,
        "global_mean":   float(y.mean()),
    }
    with open(output_path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\n[train] Modèle sauvegardé → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.output)
