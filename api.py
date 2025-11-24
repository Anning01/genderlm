# -*- coding: utf-8 -*-
import io
import base64
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image

from fairface import FairFacePredictor
from face_detector import FaceDetector

app = FastAPI(title="GenderLM API", version="3.0.0")

# Load Models
MODEL_PATH = "models/res34_fair_align_multi_7_20190809.pt"
print("Loading FairFace model...")
predictor = FairFacePredictor(MODEL_PATH)
print("FairFace model loaded.")

print("Loading Face Detector...")
try:
    face_detector = FaceDetector()
    FACE_DETECTION_AVAILABLE = True
    print("Face Detector loaded.")
except Exception as e:
    print(f"Face Detector failed to load: {e}")
    face_detector = None
    FACE_DETECTION_AVAILABLE = False

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {
            "fairface": "loaded",
            "face_detector": "available" if FACE_DETECTION_AVAILABLE else "unavailable"
        }
    }

def process_image(image: Image.Image, use_face_detection: bool, return_face_image: bool):
    results = []
    
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        # Detect faces
        cropped_images, faces = face_detector.detect_and_crop_all(image, use_bbox=True, scale=1.2)
        
        if not faces:
            # No faces found
            return {
                "face_count": 0,
                "faces": [],
                "message": "No faces detected"
            }
            
        for i, crop_img in enumerate(cropped_images):
            pred = predictor.predict(crop_img)
            face_res = {
                "face_index": i,
                "bbox": faces[i].bbox.tolist(),
                "prediction": pred
            }
            if return_face_image:
                face_res["base64"] = image_to_base64(crop_img)
            results.append(face_res)

        return {
            "face_count": len(faces),
            "faces": results
        }
    else:
        # Direct prediction
        pred = predictor.predict(image)
        return {
            "face_count": 1, # Treat whole image as one face/person
            "faces": [{
                "face_index": 0,
                "bbox": [0, 0, image.width, image.height],
                "prediction": pred
            }]
        }

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    use_face_detection: bool = Form(False),
    return_face_image: bool = Form(False)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = process_image(image, use_face_detection, return_face_image)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_mult")
def predict_mult(
    files: List[UploadFile] = File(...),
    use_face_detection: bool = Form(False),
    return_face_image: bool = Form(False)
):
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 images allowed")
        
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
             results.append({
                 "filename": file.filename,
                 "error": "Not an image file"
             })
             continue
             
        try:
            contents = file.file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            res = process_image(image, use_face_detection, return_face_image)
            res["filename"] = file.filename
            results.append(res)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
            
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
