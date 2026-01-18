# app.py
import io
import base64
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from ultralytics import YOLO

# 1) Définis le schéma de la requête
class ImageRequest(BaseModel):
    image: str  # base64-encoded

# 2) Initialise FastAPI et le modèle
app = FastAPI(title="Pothole Detection API")

# Autorise Flutter à appeler depuis n'importe où
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)




# Charge modèle une seule fois en mémoire
model = YOLO("best.pt")  

# 3) Endpoint /detect
@app.post("/detect")
async def detect_pothole(req: ImageRequest) -> Any:
    try:
        # a) Décodage base64 → PIL Image
        img_data = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        # b) Inference
        results = model.predict(np.array(img), imgsz=640, conf=0.25)
        # c) Analyse des résultats : on considère un « pothole » si  
        #    le modèle a détecté au moins une boîte avec la classe 0 (ou l’index correct)
        detections = results[0]
        has_pothole = len(detections.boxes.xyxy) > 0
        return {"potholeDetected": has_pothole}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

# 4) Lancer le serveur
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_:app", host="0.0.0.0", port=port, log_level="info")
