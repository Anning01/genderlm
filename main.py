import argparse
import os
import sys
import json
from PIL import Image
from fairface import FairFacePredictor
from face_detector import FaceDetector

def setup_models():
    MODEL_PATH = "models/res34_fair_align_multi_7_20190809.pt"
    print(f"Loading FairFace model from {MODEL_PATH}...", file=sys.stderr)
    predictor = FairFacePredictor(MODEL_PATH)
    
    try:
        face_detector = FaceDetector(det_thresh=0.4)
        face_detection_available = True
        print("Face Detector loaded.", file=sys.stderr)
    except Exception as e:
        face_detector = None
        face_detection_available = False
        print(f"Face Detector warning: {e}", file=sys.stderr)
        
    return predictor, face_detector, face_detection_available

def predict_image(image_path, predictor, face_detector, use_face_detection):
    if not os.path.exists(image_path):
        return {"error": "File not found"}
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {e}"}

    results = {
        "filename": image_path,
        "faces": []
    }

    if use_face_detection and face_detector:
        cropped_images, faces = face_detector.detect_and_crop_all(image, use_bbox=True, scale=1.2)
        if not faces:
             results["message"] = "No faces detected"
        else:
            for i, crop_img in enumerate(cropped_images):
                pred = predictor.predict(crop_img)
                results["faces"].append({
                    "face_index": i,
                    "bbox": faces[i].bbox.tolist(),
                    "prediction": pred
                })
            results["face_count"] = len(faces)
    else:
        pred = predictor.predict(image)
        results["faces"].append({
            "face_index": 0,
            "prediction": pred
        })
        results["face_count"] = 1

    return results

def main():
    parser = argparse.ArgumentParser(description="GenderLM CLI")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--crop", action="store_true", help="Enable face detection and cropping")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    predictor, face_detector, face_detection_available = setup_models()
    
    if args.crop and not face_detection_available:
        print("Warning: Face detection requested but not available. Proceeding without it.", file=sys.stderr)
        use_crop = False
    else:
        use_crop = args.crop
        
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        all_results = []
        for f in files:
            res = predict_image(f, predictor, face_detector, use_crop)
            all_results.append(res)
            
        if args.json:
            print(json.dumps(all_results, indent=2))
        else:
            for res in all_results:
                print(f"File: {res['filename']}")
                if "error" in res:
                    print(f"  Error: {res['error']}")
                elif "message" in res:
                    print(f"  {res['message']}")
                else:
                    for face in res["faces"]:
                        p = face["prediction"]
                        print(f"  Face {face.get('face_index', 0)}: {p['gender']}, {p['age']}, {p['race']}")
    else:
        res = predict_image(args.input, predictor, face_detector, use_crop)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            if "error" in res:
                print(f"Error: {res['error']}")
            elif "message" in res:
                print(res["message"])
            else:
                for face in res["faces"]:
                    p = face["prediction"]
                    print(f"Gender: {p['gender']} ({p['gender_confidence']:.2f})")
                    print(f"Age: {p['age']} ({p['age_confidence']:.2f})")
                    print(f"Race: {p['race']} ({p['race_confidence']:.2f})")

if __name__ == "__main__":
    main()
