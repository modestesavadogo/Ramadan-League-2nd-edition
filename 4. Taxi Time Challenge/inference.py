"""
inference.py — Taxi Trip Duration (v3)
========================================
Charge model.pkl, applique le feature engineering sur de nouvelles données,
génère les prédictions et les sauvegarde dans un CSV.

Usage:
    python inference.py --data test.csv [--model model.pkl] [--output predictions.csv]
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# Feature engineering (identique à train.py)
# ─────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, global_mean: float = 800.0) -> pd.DataFrame:
    """
    Feature engineering complet.
    Si trip_duration est absent (test set), les features expanding par groupe
    sont remplacées par la moyenne globale du train (stockée dans model.pkl).
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

    # ── Lag & Rolling ─────────────────────────────────────────
    gm = (df["trip_duration"].mean()
          if "trip_duration" in df.columns else global_mean)

    s = (df["trip_duration"].shift(1)
         if "trip_duration" in df.columns
         else pd.Series(gm, index=df.index))

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
        # Test set sans cible : fallback sur moyenne train
        for col in ["vendor_expand_mean", "hour_slot_mean", "dow_mean"]:
            df[col] = gm
        df["vendor_expand_std"] = 0
        df["vendor_speed_mean"] = df["distance_km"] / gm

    return df


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict(data_path: str,
            model_path: str = "model.pkl",
            output_path: str = "predictions.csv") -> pd.DataFrame:
    """
    Charge le modèle, featurise les données, génère et sauvegarde les prédictions.

    Paramètres
    ----------
    data_path   : CSV avec le même schéma que le train (trip_duration optionnel)
    model_path  : Pickle produit par train.py
    output_path : CSV de sortie avec colonne trip_duration_predicted

    Retourne
    --------
    DataFrame avec id + prédictions (+ vérité terrain si disponible)
    """
    # Chargement modèle
    print(f"[inference] Chargement modèle : {model_path}")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    model       = artefact["model"]
    features    = artefact["features"]
    global_mean = artefact.get("global_mean", 800.0)

    # Chargement & featurisation
    print(f"[inference] Chargement données : {data_path}")
    df = pd.read_csv(data_path, index_col=0)
    df = build_features(df, global_mean=global_mean)

    # Prédiction
    X = df[features]
    df["trip_duration_predicted"] = model.predict(X).clip(0).astype(int)

    # Métriques si vérité terrain disponible
    if "trip_duration" in df.columns:
        y_true = df["trip_duration"]
        y_pred = df["trip_duration_predicted"]
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print("\n── Évaluation (vérité terrain disponible) ────────")
        print(f"  MAE  = {mae:>8.1f} s  (~{mae/60:.1f} min)")
        print(f"  RMSE = {rmse:>8.1f} s  (~{rmse/60:.1f} min)")
        print(f"  R²   = {r2:>8.4f}")
        print("──────────────────────────────────────────────────")

    # Export
    out_cols = ["id", "trip_duration_predicted"]
    if "trip_duration" in df.columns:
        out_cols.append("trip_duration")
    df[out_cols].to_csv(output_path, index=False)

    print(f"\n[inference] Prédictions sauvegardées → {output_path}")
    print(f"[inference] Aperçu :\n{df[out_cols].head(5).to_string(index=False)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--model",  default="model.pkl")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(args.data, args.model, args.output)
