from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from skmultilearn.ensemble import RakelD
import joblib
import io
from pydantic import BaseModel

# Initialisation de l'application
app = FastAPI(title="API Classification Dommages Auto")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Chargement des modèles au démarrage
print("Chargement des modèles...")
try:
    feature_extractor = tf.keras.models.load_model('Assurance_efficientnet_feature_extractor.h5')
    classifier = joblib.load('Assurance_classifier.pkl')
    print("Modèles chargés avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {str(e)}")
    raise e

# Configuration des labels
LABEL_NAMES = [
    "bonnet-dent",
    "doorouter-dent",
    "fender-dent",
    "front-bumper-dent",
    "rear-bumper-dent"
]

# Modèle de réponse
class PredictionResponse(BaseModel):
    labels: list[str]
    probabilities: list[float]
    success: bool

def process_image(image_bytes: bytes) -> np.ndarray:
    """Prétraitement de l'image"""
    try:
        # Conversion bytes -> numpy array
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Suppression de bruit
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Conversion BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionnement et normalisation
        img = cv2.resize(img, (224, 224))
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement d'image: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_damage(file: UploadFile = File(...)):
    """Endpoint de prédiction"""
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")
        
        # Lecture et prétraitement
        image_bytes = await file.read()
        processed_img = process_image(image_bytes)
        
        # Extraction des caractéristiques
        features = feature_extractor.predict(processed_img)
        features = features.reshape(1, -1)
        
        # Prédiction
        probas = classifier.predict_proba(features).toarray()[0]
        predictions = classifier.predict(features).toarray()[0]
        
        # Formatage des résultats
        result_labels = [LABEL_NAMES[i] for i, pred in enumerate(predictions) if pred == 1]
        result_probas = [float(probas[i]) for i, pred in enumerate(predictions) if pred == 1]
        
        return {
            "labels": result_labels,
            "probabilities": result_probas,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Endpoint de vérification de santé"""
    return {"status": "healthy", "models_loaded": True}