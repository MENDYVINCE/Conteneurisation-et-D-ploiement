# from models.lstm_ae import build_models
from models.gru_ae import build_models # for docker

import pandas as pd
import numpy as np
from keras import regularizers, layers, models, callbacks
from sklearn.metrics import f1_score, precision_score, recall_score
import keras

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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


def get_X_y(df: pd.DataFrame) -> tuple:
    logger.info("Séparation des features (X) et de la target (y)")
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Distribution de y - Normal: {(y==0).sum()}, Anomalie: {(y==1).sum()}")

    joblib.dump(list(X.columns), "artifacts/features.pkl")

    return X, y


def get_tvt_data(X, y) -> tuple:
    logger.info("Séparation des données en train/val/test")
    n_total = len(X)

    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)

    X_train_full = X[:n_train]
    y_train_full = y[:n_train]
    X_train = X_train_full[y_train_full == 0]
    y_train = y_train_full[y_train_full == 0]

    X_val_full = X[n_train : n_train + n_val]
    y_val_full = y[n_train : n_train + n_val]
    X_val = X_val_full[y_val_full == 0]
    y_val = y_val_full[y_val_full == 0]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    logger.info(f"Train: {len(X_train)} échantillons (seulement normaux)")
    logger.info(f"Validation: {len(X_val)} échantillons (seulement normaux)")
    logger.info(
        f"Test: {len(X_test)} échantillons (normal: {(y_test==0).sum()}, anomalie: {(y_test==1).sum()})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    logger.info("Normalisation des données avec MinMaxScaler")
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    logger.info("Scaler sauvegardé dans artifacts/scaler.pkl")
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test


def create_sequences(X, timesteps):
    logger.info(f"Création de séquences avec timesteps={timesteps}")
    sequences = []
    for i in range(len(X) - timesteps + 1):
        sequences.append(X[i : i + timesteps])
    sequences_array = np.array(sequences)
    logger.info(f"Shape des séquences: {sequences_array.shape}")
    return sequences_array


def optimize_threshold_with_sequences(autoencoder, X_test, y_test, timesteps):
    logger.info("Optimisation du seuil de détection")

    # 1. Créer les séquences pour X_test et y_test
    X_test_seq = create_sequences(X_test, timesteps)

    # Labelliser les séquences : 1 si au moins une anomalie dans la séquence
    y_test_seq = []
    for i in range(len(y_test) - timesteps + 1):
        sequence_labels = y_test[i : i + timesteps]
        y_test_seq.append(1 if np.any(sequence_labels == 1) else 0)
    y_test_seq = np.array(y_test_seq)

    logger.info(
        f"Séquences de test - Normal: {(y_test_seq==0).sum()}, Anomalie: {(y_test_seq==1).sum()}"
    )

    # 2. Prédire les reconstructions pour TOUTES les séquences
    logger.info("Prédiction des reconstructions...")
    reconstructions = autoencoder.predict(X_test_seq)

    # 3. Calculer le MSE par séquence
    mse = np.mean(np.square(X_test_seq - reconstructions), axis=(1, 2))

    # 4. Séparer les erreurs pour les séquences normales et anormales
    mse_normal = mse[y_test_seq == 0]
    mse_anomaly = mse[y_test_seq == 1]

    logger.info(
        f"MSE moyen (normal): {np.mean(mse_normal):.4f} (std: {np.std(mse_normal):.4f})"
    )
    logger.info(
        f"MSE moyen (anomaly): {np.mean(mse_anomaly):.4f} (std: {np.std(mse_anomaly):.4f})"
    )

    # 5. Tester différents seuils pour maximiser le F1-score
    logger.info("Recherche du meilleur seuil...")
    thresholds = np.linspace(min(mse_normal), max(mse_anomaly), 100)
    best_f1 = 0
    best_seuil = 0
    best_precision = 0
    best_recall = 0

    for seuil in thresholds:
        pred = (mse > seuil).astype(int)
        f1 = f1_score(y_test_seq, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_seuil = seuil
            best_precision = precision_score(y_test_seq, pred)
            best_recall = recall_score(y_test_seq, pred)

    metrics = {"f1": best_f1, "precision": best_precision, "recall": best_recall}

    joblib.dump(metrics, "artifacts/metrics.pkl")

    joblib.dump(best_seuil, "artifacts/threshold.pkl")

    logger.info("=" * 60)
    logger.info(f"MEILLEUR SEUIL: {best_seuil:.4f}")
    logger.info(f"F1-Score: {best_f1:.3f}")
    logger.info(f"Précision: {best_precision:.3f}")
    logger.info(f"Rappel: {best_recall:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE")
    logger.info("=" * 60)

    # Constantes
    CORRELATION_THRESHOLD = 0.92
    TIMESTEPS = 10  # 20
    N_EPOCHS = 100
    BATCH_SIZE = 32
    N_PATIENCE = 5

    logger.info(f"Paramètres:")
    logger.info(f"  - CORRELATION_THRESHOLD: {CORRELATION_THRESHOLD}")
    logger.info(f"  - TIMESTEPS: {TIMESTEPS}")
    logger.info(f"  - N_EPOCHS: {N_EPOCHS}")
    logger.info(f"  - BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  - N_PATIENCE: {N_PATIENCE}")
    logger.info("=" * 60)

    # Chargement des données
    logger.info("Chargement des données depuis datas/ai4i2020.csv")
    df = pd.read_csv("datas/ai4i2020.csv")
    logger.info(f"Données chargées - Shape: {df.shape}")

    # Préparation des données
    df = preprocess_data(df)
    df = build_features(df)
    df = clean_data(df, correlation_threshold=CORRELATION_THRESHOLD)
    X, y = get_X_y(df)
    X_train, X_val, X_test, _, _, y_test = get_tvt_data(X, y)

    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    X_train_seq = create_sequences(X_train, timesteps=TIMESTEPS)
    X_val_seq = create_sequences(X_val, timesteps=TIMESTEPS)

    # Construction et compilation du modèle
    encoder, decoder, autoencoder = build_models(
        timesteps=TIMESTEPS, n_features=X_train_seq.shape[2]
    )

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=N_PATIENCE,
        min_lr=1e-6,
        verbose=1,
    )

    # Entraînement
    logger.info("=" * 60)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT")
    logger.info("=" * 60)

    history = autoencoder.fit(
        X_train_seq,
        X_train_seq,
        validation_data=(X_val_seq, X_val_seq),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        shuffle=False,
    )

    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT TERMINÉ")
    logger.info("=" * 60)

    # Optimisation du seuil
    seuil = optimize_threshold_with_sequences(
        autoencoder=autoencoder, X_test=X_test, y_test=y_test, timesteps=TIMESTEPS
    )

    # Sauvegarde des modèles
    logger.info("Sauvegarde des modèles...")
    autoencoder.save("artifacts/autoencoder.keras")
    logger.info("Autoencoder sauvegardé dans artifacts/autoencoder.keras")
    encoder.save("artifacts/encoder.keras")
    logger.info("Encoder sauvegardé dans artifacts/encoder.keras")

    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)
