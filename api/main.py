import numpy as np
from fastapi import FastAPI, HTTPException
from api.schemas.input_schema import CustomerFeatures, FEATURE_ORDER
from api.model.loader import load_model
from api.utils.business_cost import COST_FN, COST_FP
from api.utils.logging import log_prediction
from api.model.preprocess import preprocess_X


app = FastAPI(
    title="Credit Scoring API",
    description="API pour prédire le risque client",
    version="1.0.0"
)

# seuil optimal récupéré depuis MLflow
THRESHOLD = 0.42  


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        X = np.array([[features.dict()[f] for f in FEATURE_ORDER]])
        print("📥 Reçu features:", features.dict())
        print("🔧 Ordered list:", [features.dict().get(f) for f in FEATURE_ORDER])
        X = preprocess_X(X)  # 👉 imputation automatique
        # 🔧 Support du dummy model pendant les tests
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
        else:
            # Mode test : le dummy retourne déjà une classe 0 ou 1
            pred_class = model.predict(X)[0]
            proba = float(pred_class)  # probabilité simulée
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

    pred = int(proba >= THRESHOLD)

    log_prediction(features.dict(), float(proba), pred)

    return {
        "probability_default": float(proba),
        "prediction": pred,
        "threshold_used": THRESHOLD,
        "business_cost_FN": COST_FN,
        "business_cost_FP": COST_FP
    }


