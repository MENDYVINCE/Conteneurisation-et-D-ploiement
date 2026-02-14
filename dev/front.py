import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Détection d'Anomalies", layout="centered")
st.title("Détection d'Anomalies - Autoencoder LSTM")

# Upload CSV
uploaded_file = st.file_uploader("Choisis ton CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.head(3000)
    st.write("Aperçu des données :")
    st.dataframe(df.head())

    if st.button("Lancer la détection"):
        with st.spinner("Prédiction en cours..."):
            # Préparer le JSON pour l'API
            json_data = {"rows": df.to_dict(orient="records")}

            # st.subheader("Aperçu du JSON envoyé")
            # st.json(json_data["rows"][:3])

            try:
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    json=json_data
                )
                response.raise_for_status()
                data = response.json()

                st.success(f"{data['n_predictions']} lignes analysées")

                st.subheader("Performances du modèle")
                st.json(data["model_performance"])

                st.subheader("Résultats (anomalies détectées)")
                results_df = pd.DataFrame(data["results"])

                # Ajouter Product ID depuis les données d'entrée
                results_df["Product ID"] = df["Product ID"]

                # Compter le nombre d'anomalies (supposons que la colonne 'anomaly' existe dans results)
                if "anomaly" in results_df.columns:
                    n_anomalies = results_df["anomaly"].sum()
                else:
                    # Si ton predictor retourne 1 pour anomalie dans chaque ligne
                    n_anomalies = len(results_df)  # ou adapte selon ton retour exact

                st.metric("Nombre total d'anomalies détectées", n_anomalies)

                # Afficher le DataFrame dans un expand pour exploration
                with st.expander("Voir le DataFrame complet des anomalies"):
                    st.dataframe(results_df.set_index("Product ID"))


            except requests.exceptions.HTTPError as err:
                st.error(f"HTTP error occurred: {err}")
            except Exception as err:
                st.error(f"An error occurred: {err}")
