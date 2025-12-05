import os
import joblib

# Détection du mode test (GitHub Actions)
TESTING = os.getenv("TESTING") == "1"

LOCAL_MODEL_PATH = "model/model.pkl"

model = None


def load_model():
    """
    Charge le modèle utilisé par l'API.
    - En mode TESTING (GitHub Actions) → DummyModel pour éviter les dépendances lourdes.
    - En production / local → Chargement du modèle .pkl.
    """

    global model

    # Si un modèle est déjà chargé, ne pas recharger
    if model is not None:
        return model

    # 🧪 MODE TEST : retourne un modèle factice compatible predict_proba
    if TESTING:
        print("🧪 Mode TESTING détecté — utilisation d’un modèle factice.")
        class DummyModel:
            def predict_proba(self, X):
                # renvoie probabilité 0.1 de défaut pour éviter erreurs
                return [[0.9, 0.1]]
        model = DummyModel()
        return model

    # 🗃️ MODE NORMAL → charger le modèle local
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
