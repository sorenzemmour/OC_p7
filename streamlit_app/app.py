import streamlit as st
import requests
import numpy as np
import pandas as pd

# 👉 À remplacer par ton URL Render
API_URL = "https://oc-p7-3.onrender.com/"

st.set_page_config(page_title="Credit Scoring", layout="centered")

st.title("📊 Crédit Scoring — Interface Client")
st.write("Cette interface envoie les caractéristiques du client à l’API et récupère la prédiction.")


# ----- FORMULAIRE DES FEATURES -----
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

# ----- Upload CSV -----
st.markdown("---")
st.subheader("📁 Prédictions à partir d’un fichier CSV")

uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    import pandas as pd

    df = pd.read_csv(uploaded_file)

    st.write("Aperçu du fichier :")
    st.dataframe(df.head())

    # Vérification des colonnes requises
    required_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "REG_CITY_NOT_WORK_CITY", "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY", "DAYS_EMPLOYED",
        "DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Le CSV ne contient pas toutes les colonnes requises.")
    else:
        if st.button("📥 Lancer les prédictions CSV"):
            results = []
            with st.spinner("Prédictions en cours..."):
                for _, row in df.iterrows():
                    payload = row[required_cols].to_dict()

                    # Nettoyage JSON-friendly
                    for key, value in payload.items():
                        # Remplace NaN par None
                        if pd.isna(value):
                            payload[key] = None

                        # Convertit numpy types → Python types
                        elif isinstance(value, np.generic):
                            payload[key] = value.item()

                    try:
                        res = requests.post(API_URL, json=payload)
                        
                        if res.status_code == 200:
                            results.append(res.json())
                        else:
                            results.append({"error": f"API error {res.status_code}"})

                    except Exception as e:
                        results.append({"error": str(e)})

            results_df = pd.DataFrame(results)
            st.success("🎉 Prédictions terminées !")
            st.dataframe(results_df)

            # Export CSV des résultats
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📤 Télécharger les résultats",
                data=csv_export,
                file_name="predictions.csv",
                mime="text/csv",
            )



# ------ BOUTON DE PRÉDICTION ------
if st.button("📥 Obtenir la prédiction"):
    payload = {
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "REG_CITY_NOT_WORK_CITY": REG_CITY_NOT_WORK_CITY,
        "REGION_RATING_CLIENT": REGION_RATING_CLIENT,
        "REGION_RATING_CLIENT_W_CITY": REGION_RATING_CLIENT_W_CITY,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "DAYS_BIRTH": DAYS_BIRTH,
        "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
        "DAYS_LAST_PHONE_CHANGE": DAYS_LAST_PHONE_CHANGE
    }

    with st.spinner("Envoi de la requête à l’API..."):
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()

                st.success("🎉 Prédiction reçue !")
                
                st.metric("Probabilité de défaut", f"{data['probability_default']:.2%}")
                
                prediction = "❌ Risque élevé" if data["prediction"] == 1 else "✔️ Risque faible"
                st.write("### Résultat :", prediction)

                st.write(f"Seuil utilisé : **{data['threshold_used']}**")
                st.write(f"Coût FN : **{data['business_cost_FN']}**")
                st.write(f"Coût FP : **{data['business_cost_FP']}**")

            else:
                st.error(f"Erreur API : {res.status_code}")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
