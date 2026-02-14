import pandas as pd
import numpy as np
import joblib
import keras
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import os
import warnings

# Supprimer les warnings Python
warnings.filterwarnings("ignore")

# Suppression des logs TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=info, 2=warning, 3=error
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU uniquement

# Désactiver les messages oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Début du prétraitement des données")
    df = df.set_index(keys="UDI", drop=True)
    df = df.drop(columns=["Product ID"])
    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"]).copy()
    df = pd.get_dummies(df, columns=["Type"], prefix="Type", drop_first=True, dtype=int)
    logger.info(f"Prétraitement terminé - Shape: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Construction des features")
    df = df.copy()
    df["Power_kw"] = (df["Torque [Nm]"] * df["Rotational speed [rpm]"]) / 9550
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Speed_torque_ratio"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1)
    max_wear = df["Tool wear [min]"].max()
    df["Wear_level"] = pd.cut(
        df["Tool wear [min]"],
        bins=[0, 80, 160, max_wear + 1],
        labels=[0, 1, 2],
        include_lowest=True,
    )
    df["Wear_level"] = (
        pd.to_numeric(df["Wear_level"], errors="coerce").fillna(0).astype(int)
    )
    df["High_wear"] = (df["Tool wear [min]"] > 200).astype(int)
    df["Thermal_load"] = df["Temp_diff"] * df["Power_kw"]
    df["Mechanical_stress"] = df["Torque [Nm]"] * (1 + df["Tool wear [min]"] / 250)

    logger.info(f"Features construites - Nombre total de colonnes: {df.shape[1]}")
    return df


def clean_data(df: pd.DataFrame, correlation_threshold=0.9) -> pd.DataFrame:
    logger.info(
        f"Nettoyage des données avec seuil de corrélation: {correlation_threshold}"
    )
    corr = df.drop(columns=["Machine failure"]).corr(method="pearson")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
    )

    corr_pairs.columns = ["var1", "var2", "correlation"]
    strong_corr_pairs = corr_pairs[corr_pairs["correlation"] > correlation_threshold]

    logger.info(f"Nombre de paires fortement corrélées: {len(strong_corr_pairs)}")

    features_to_drop = []
    for _, row in strong_corr_pairs.iterrows():
        var1, var2 = row["var1"], row["var2"]
        if df[var1].var() < df[var2].var():
            features_to_drop.append(var1)
        else:
            features_to_drop.append(var2)

    df = df.drop(columns=features_to_drop)
    logger.info(f"Features supprimées: {features_to_drop}")
    logger.info(f"Shape après nettoyage: {df.shape}")

    return df


def create_sequences(X, timesteps):
    logger.info(f"Création de séquences avec timesteps={timesteps}")
    sequences = []
    for i in range(len(X) - timesteps + 1):
        sequences.append(X[i : i + timesteps])
    sequences_array = np.array(sequences)
    logger.info(f"Shape des séquences: {sequences_array.shape}")
    return sequences_array


class AnomalyPredictor:

    def __init__(self):

        self.autoencoder = keras.models.load_model("artifacts/autoencoder.keras")
        self.scaler = joblib.load("artifacts/scaler.pkl")
        self.threshold = joblib.load("artifacts/threshold.pkl")
        self.features = joblib.load("artifacts/features.pkl")
        self.metrics = joblib.load("artifacts/metrics.pkl")

        self.TIMESTEPS = 10

    def get_metrics(self):
        return self.metrics

    def preprocess(self, df):

        df = preprocess_data(df)
        df = build_features(df)
        df = clean_data(df)

        # align columns (TRÈS IMPORTANT)
        df = df.reindex(columns=self.features, fill_value=0)

        return df

    def predict(self, df_raw):

        # pipeline identique training
        df = self.preprocess(df_raw)

        X = df.values

        # scaling
        X_scaled = self.scaler.transform(X)

        # sequences
        X_seq = create_sequences(X_scaled, self.TIMESTEPS)

        # reconstruction
        recon = self.autoencoder.predict(X_seq)

        # reconstruction error
        mse = np.mean(np.square(X_seq - recon), axis=(1, 2))

        anomalies = mse > self.threshold

        results = []

        for score, flag in zip(mse, anomalies):
            results.append({"anomaly": bool(flag), "score": float(score)})

        return results
