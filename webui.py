# -*- coding: utf-8 -*-
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
from fairface import FairFacePredictor
from face_detector import FaceDetector

# Load Models
MODEL_PATH = "models/res34_fair_align_multi_7_20190809.pt"
print("Loading models...")
predictor = FairFacePredictor(MODEL_PATH)

try:
    face_detector = FaceDetector()
    FACE_DETECTION_AVAILABLE = True
except:
    face_detector = None
    FACE_DETECTION_AVAILABLE = False
print("Models loaded.")

def draw_bbox(image, bbox, label):
    draw = ImageDraw.Draw(image)
    # bbox: x1, y1, x2, y2
    draw.rectangle(bbox, outline="red", width=3)
    
    # Draw label background
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    text_size = draw.textbbox((0, 0), label, font=font)
    text_w = text_size[2] - text_size[0]
    text_h = text_size[3] - text_size[1]
    
    draw.rectangle([bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1]], fill="red")
    draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill="white", font=font)
    return image

def predict_single(image, use_face_detection):
    if image is None:
        return None, "Please upload an image.", None
    
    image = image.convert("RGB")
    info_text = ""
    output_image = image.copy()
    crops = []
    
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        cropped_images, faces = face_detector.detect_and_crop_all(image, use_bbox=True, scale=1.2)
        
        if not faces:
            info_text += "No faces detected. Predicting on whole image.\n\n"
            pred = predictor.predict(image)
            info_text += f"Whole Image:\nGender: {pred['gender']} ({pred['gender_confidence']:.2f})\nAge: {pred['age']}\nRace: {pred['race']}\n"
        else:
            info_text += f"Detected {len(faces)} faces.\n\n"
            
            # Batch predict
            predictions = predictor.predict_batch(cropped_images)
            
            for i, (pred, crop_img, face) in enumerate(zip(predictions, cropped_images, faces)):
                label = f"{pred['gender']}, {pred['age']}"
                draw_bbox(output_image, face.bbox, f"{i+1}")
                
                info_text += f"Face {i+1}:\n"
                info_text += f"  Gender: {pred['gender']} ({pred['gender_confidence']:.2f})\n"
                info_text += f"  Age: {pred['age']} ({pred['age_confidence']:.2f})\n"
                info_text += f"  Race: {pred['race']} ({pred['race_confidence']:.2f})\n"
                info_text += f"  Time: {pred['inference_time_ms']:.1f}ms\n\n"
                
                crops.append((crop_img, f"Face {i+1}"))
                
    else:
        pred = predictor.predict(image)
        info_text += "Direct Prediction (No Detection):\n"
        info_text += f"Gender: {pred['gender']} ({pred['gender_confidence']:.2f})\n"
        info_text += f"Age: {pred['age']} ({pred['age_confidence']:.2f})\n"
        info_text += f"Race: {pred['race']} ({pred['race_confidence']:.2f})\n"
        info_text += f"Time: {pred['inference_time_ms']:.1f}ms\n"

    return output_image, info_text, crops

def predict_batch_files(files, use_face_detection):
    if not files:
        return "No files selected."
        
    results_text = []
    all_crops = []
    tasks = [] # {filename, start_idx, count, faces, valid, error}
    
    # 1. Detect Faces
    for file in files:
        task = {"filename": os.path.basename(file.name), "valid": False}
        try:
            image = Image.open(file.name).convert("RGB")
            task["valid"] = True
            
            if use_face_detection and FACE_DETECTION_AVAILABLE:
                crops, faces = face_detector.detect_and_crop_all(image, use_bbox=True, scale=1.2)
                if not faces:
                    task["count"] = 0
                    task["message"] = "No faces detected"
                else:
                    task["start_idx"] = len(all_crops)
                    task["count"] = len(crops)
                    all_crops.extend(crops)
            else:
                task["start_idx"] = len(all_crops)
                task["count"] = 1
                all_crops.append(image)
                
        except Exception as e:
            task["valid"] = False
            task["error"] = str(e)
            
        tasks.append(task)
        
    # 2. Batch Predict
    if all_crops:
        all_predictions = predictor.predict_batch(all_crops)
    else:
        all_predictions = []
        
    # 3. Format Results
    for task in tasks:
        results_text.append(f"--- {task['filename']} ---")
        
        if not task["valid"]:
            results_text.append(f"Error: {task.get('error', 'Unknown error')}")
        elif "message" in task:
            results_text.append(task["message"])
        else:
            start = task["start_idx"]
            count = task["count"]
            preds = all_predictions[start : start + count]
            
            for i, pred in enumerate(preds):
                if count > 1 or use_face_detection:
                    prefix = f"Face {i+1}: "
                else:
                    prefix = "Result: "
                results_text.append(f"{prefix}{pred['gender']}, {pred['age']}, {pred['race']}")
                
        results_text.append("")
            
    return "\n".join(results_text)

with gr.Blocks(title="GenderLM FairFace WebUI") as demo:
    gr.Markdown("# GenderLM with FairFace")
    gr.Markdown("Predict Gender, Age, and Race.")
    
    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Image")
                    use_det = gr.Checkbox(label="Use Face Detection", value=True, interactive=FACE_DETECTION_AVAILABLE)
                    btn = gr.Button("Predict", variant="primary")
                with gr.Column():
                    out_img = gr.Image(type="pil", label="Annotated Image")
                    out_text = gr.Textbox(label="Results", lines=10)
                    out_gallery = gr.Gallery(label="Detected Faces")
            
            btn.click(predict_single, inputs=[input_img, use_det], outputs=[out_img, out_text, out_gallery])
            
        with gr.Tab("Batch Processing"):
            files = gr.File(file_count="multiple", label="Upload Images")
            batch_use_det = gr.Checkbox(label="Use Face Detection", value=True, interactive=FACE_DETECTION_AVAILABLE)
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_out = gr.Textbox(label="Batch Results", lines=20)
            
            batch_btn.click(predict_batch_files, inputs=[files, batch_use_det], outputs=[batch_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
