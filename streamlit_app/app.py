import streamlit as st
import requests
import numpy as np
import pandas as pd

# 👉 URL de ton API FastAPI sur Render (avec /predict à la fin)
API_URL = "https://oc-p7-4.onrender.com/predict"

# Features attendues par l'API (schéma Pydantic)
REQUIRED_FEATURES = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "REG_CITY_NOT_WORK_CITY",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
]

INT_FEATURES = {
    "REG_CITY_NOT_WORK_CITY",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "DAYS_ID_PUBLISH",
    "DAYS_BIRTH",
}


def build_payload_from_row(row: pd.Series):
    """
    Construit un payload JSON propre pour l'API à partir d'une ligne de DataFrame.
    - vérifie les colonnes obligatoires
    - gère NaN
    - convertit les types numpy → types natifs Python
    - caste les features entières en int
    """
    payload = {}

    for feat in REQUIRED_FEATURES:
        if feat not in row.index:
            return None, f"Colonne manquante dans le CSV : {feat}"

        value = row[feat]

        # NaN → rejet de la ligne (l'API ne veut pas de null sur champs requis)
        if pd.isna(value):
            return None, f"Valeur manquante pour la feature {feat}"

        # numpy types → types natifs
        if isinstance(value, np.generic):
            value = value.item()

        # Cast explicite des entiers
        if feat in INT_FEATURES:
            try:
                value = int(value)
            except Exception:
                return None, f"Impossible de convertir {feat} en int (valeur={value})"
        else:
            # float "classique"
            try:
                value = float(value)
            except Exception:
                return None, f"Impossible de convertir {feat} en float (valeur={value})"

        payload[feat] = value

    return payload, None


# -------------------------------------------------------------------
# CONFIG STREAMLIT
# -------------------------------------------------------------------
st.set_page_config(page_title="Credit Scoring", layout="centered")

st.title("📊 Crédit Scoring — Interface Client")
st.write(
    "Cette interface envoie les caractéristiques du client à l’API FastAPI "
    "et récupère la probabilité de défaut ainsi que la décision binaire."
)

# -------------------------------------------------------------------
# FORMULAIRE MANUEL
# -------------------------------------------------------------------
st.subheader("🧮 Entrer les caractéristiques du client")

col1, col2 = st.columns(2)

with col1:
    EXT_SOURCE_1 = st.number_input("EXT_SOURCE_1", min_value=0.0, max_value=1.0, step=0.01)
    EXT_SOURCE_2 = st.number_input("EXT_SOURCE_2", min_value=0.0, max_value=1.0, step=0.01)
    EXT_SOURCE_3 = st.number_input("EXT_SOURCE_3", min_value=0.0, max_value=1.0, step=0.01)
    REG_CITY_NOT_WORK_CITY = st.selectbox("REG_CITY_NOT_WORK_CITY", [0, 1])

with col2:
    REGION_RATING_CLIENT = st.number_input("REGION_RATING_CLIENT", min_value=0, max_value=3, step=1)
    REGION_RATING_CLIENT_W_CITY = st.number_input("REGION_RATING_CLIENT_W_CITY", min_value=0, max_value=3, step=1)
    DAYS_EMPLOYED = st.number_input("DAYS_EMPLOYED", value=-2000)
    DAYS_BIRTH = st.number_input("DAYS_BIRTH", value=-12000)
    DAYS_ID_PUBLISH = st.number_input("DAYS_ID_PUBLISH", value=-500)
    DAYS_LAST_PHONE_CHANGE = st.number_input("DAYS_LAST_PHONE_CHANGE", value=-300.0)

# ------ BOUTON DE PRÉDICTION (manuel) ------
if st.button("📥 Obtenir la prédiction"):
    manual_row = pd.Series(
        {
            "EXT_SOURCE_1": EXT_SOURCE_1,
            "EXT_SOURCE_2": EXT_SOURCE_2,
            "EXT_SOURCE_3": EXT_SOURCE_3,
            "REG_CITY_NOT_WORK_CITY": REG_CITY_NOT_WORK_CITY,
            "REGION_RATING_CLIENT": REGION_RATING_CLIENT,
            "REGION_RATING_CLIENT_W_CITY": REGION_RATING_CLIENT_W_CITY,
            "DAYS_EMPLOYED": DAYS_EMPLOYED,
            "DAYS_BIRTH": DAYS_BIRTH,
            "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
            "DAYS_LAST_PHONE_CHANGE": DAYS_LAST_PHONE_CHANGE,
        }
    )

    payload, err = build_payload_from_row(manual_row)

    if err:
        st.error(f"❌ Erreur de validation locale : {err}")
    else:
        with st.spinner("Envoi de la requête à l’API..."):
            try:
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    data = res.json()

                    st.success("🎉 Prédiction reçue !")

                    st.metric("Probabilité de défaut", f"{data['probability_default']:.2%}")

                    prediction = (
                        "❌ Risque élevé" if data["prediction"] == 1 else "✔️ Risque faible"
                    )
                    st.write("### Résultat :", prediction)

                    st.write(f"Seuil utilisé : **{data['threshold_used']}**")
                    st.write(f"Coût FN : **{data['business_cost_FN']}**")
                    st.write(f"Coût FP : **{data['business_cost_FP']}**")
                else:
                    st.error(f"Erreur API : {res.status_code} — {res.text}")

            except Exception as e:
                st.error(f"❌ Erreur de requête : {e}")


# -------------------------------------------------------------------
# UPLOAD CSV
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📁 Prédictions à partir d’un fichier CSV")

uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Aperçu du fichier :")
    st.dataframe(df.head())

    # Vérifie que toutes les colonnes nécessaires existent au global
    missing_global = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing_global:
        st.error(
            "❌ Le CSV ne contient pas toutes les colonnes requises : "
            + ", ".join(missing_global)
        )
    else:
        if st.button("📥 Lancer les prédictions CSV"):
            results = []
            invalid_rows = []

            with st.spinner("Prédictions en cours..."):
                for idx, row in df.iterrows():
                    payload, err = build_payload_from_row(row)

                    # Si la ligne est invalide localement, on ne l’envoie pas à l’API
                    if err:
                        invalid_rows.append(
                            {"index": idx, "SK_ID_CURR": row.get("SK_ID_CURR", None), "error": err}
                        )
                        results.append(
                            {
                                "index": idx,
                                "SK_ID_CURR": row.get("SK_ID_CURR", None),
                                "error": f"Validation locale : {err}",
                            }
                        )
                        continue

                    try:
                        res = requests.post(API_URL, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            data["index"] = idx
                            data["SK_ID_CURR"] = row.get("SK_ID_CURR", None)
                            results.append(data)
                        else:
                            results.append(
                                {
                                    "index": idx,
                                    "SK_ID_CURR": row.get("SK_ID_CURR", None),
                                    "error": f"API error {res.status_code}: {res.text}",
                                }
                            )
                    except Exception as e:
                        results.append(
                            {
                                "index": idx,
                                "SK_ID_CURR": row.get("SK_ID_CURR", None),
                                "error": f"Exception: {e}",
                            }
                        )

            results_df = pd.DataFrame(results)
            st.success("🎉 Prédictions terminées !")
            st.dataframe(results_df)

            if invalid_rows:
                st.warning(
                    f"{len(invalid_rows)} lignes n'ont pas été envoyées à l’API "
                    "car elles contenaient des valeurs manquantes ou invalides sur les features requises."
                )
                st.write(pd.DataFrame(invalid_rows))

            # Export CSV des résultats
            csv_export = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📤 Télécharger les résultats",
                data=csv_export,
                file_name="predictions.csv",
                mime="text/csv",
            )
