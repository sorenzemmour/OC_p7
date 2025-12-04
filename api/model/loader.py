import os
import joblib

# Détection du mode test
TESTING = os.getenv("TESTING") == "1"

if not TESTING:
    import mlflow
    import mlflow.sklearn

RUN_ID = "220b6b0558b049688b2ece173f794542"
MODEL_URI = f"runs:/{RUN_ID}/model"

LOCAL_MODEL_PATH = "model/model.pkl"

model = None


def load_model():
    global model

    # Si un modèle est déjà chargé, le renvoyer
    if model is not None:
        return model

    # 🧪 MODE TEST : renvoie un dummy model simple
    if TESTING:
        print("🧪 Mode TESTING détecté — utilisation d’un modèle factice.")
        class DummyModel:
            def predict(self, X):
                return [0]  # valeur stable
        model = DummyModel()
        return model

    # 🖥️ MODE NORMAL (local / prod) → essayer MLflow
    try:
        print("🔄 Tentative de chargement via MLflow...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ Modèle chargé depuis MLflow.")
        return model
    except Exception as e:
        print(f"⚠️ MLflow indisponible : {e}")

    # 🗃️ Fallback : modèle local
    try:
        print("🔄 Chargement du modèle local...")
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"Fichier {LOCAL_MODEL_PATH} introuvable")

        model = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Modèle local chargé.")
        return model
    except Exception as e:
        print(f"❌ ERREUR — Impossible de charger le modèle local : {e}")
        raise RuntimeError("Aucun modèle disponible pour l'inférence.")
