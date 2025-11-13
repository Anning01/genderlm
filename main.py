from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import sys
import os
from pathlib import Path

# 模型路径配置
LOCAL_MODEL_PATH = "./gender-classification-2"
HF_MODEL_NAME = "rizvandwiki/gender-classification-2"

# 优先使用本地模型，如果不存在则从 Hugging Face 下载
if os.path.exists(LOCAL_MODEL_PATH):
    print(f"✅ 使用本地模型: {LOCAL_MODEL_PATH}")
    model_path = LOCAL_MODEL_PATH
else:
    print(f"⬇️  本地模型不存在，从 Hugging Face 下载: {HF_MODEL_NAME}")
    model_path = HF_MODEL_NAME

# 加载图像处理器和模型
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

def predict_gender(image_path: str):
    # 打开图片
    image = Image.open(image_path).convert("RGB")

    # 预处理
    inputs = processor(images=image, return_tensors="pt")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()

    # 取出类别标签
    label = model.config.id2label[predicted_class_id]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_id].item()

    print(f"✅ Image: {image_path}")
    print(f"Predicted gender: {label} ({confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gender_predict.py <image_path>")
    else:
        predict_gender(sys.argv[1])
