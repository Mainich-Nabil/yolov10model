# app_.py
import io
import base64
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# --------------------------
# 1) Request schema
# --------------------------
class ImageRequest(BaseModel):
    image: str  # base64-encoded

# --------------------------
# 2) Initialize FastAPI
# --------------------------
app = FastAPI(title="Pothole Detection API")

# Allow requests from anywhere (Flutter / mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --------------------------
# 3) Load YOLO model safely (keep full checkpoint)
# --------------------------
with torch.serialization.add_safe_globals([DetectionModel]):
    model = YOLO("best.pt")  # keeps everything unchanged

# --------------------------
# 4) Test endpoint
# --------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# --------------------------
# 5) Pothole detection endpoint
# --------------------------
@app.post("/detect")
async def detect_pothole(req: ImageRequest) -> Any:
    try:
        # Decode base64 → PIL Image
        img_data = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Run inference
        results = model.predict(np.array(img), imgsz=640, conf=0.25)

        # Consider a pothole detected if at least 1 box
        detections = results[0]
        has_pothole = len(detections.boxes.xyxy) > 0

        return {"potholeDetected": has_pothole}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# --------------------------
# 6) Start server
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("app_:app", host="0.0.0.0", port=port, workers=1)
