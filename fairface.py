import time
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
import PIL.Image as Image

class FairFacePredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.load_model(model_path, self.device)
        
        # Labels
        self.race_labels = ['White', 'Black', 'Latino', 'Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.gender_labels = ['Female', 'Male']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Warmup
        print("FairFace Model loaded. Warming up...")
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)
        print("Warmup done.")

    def load_model(self, model_path, device):
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def preprocess_batch(self, images):
        """
        Preprocess a list of images.
        """
        tensors = []
        for img in images:
            # Handle PIL Image
            if isinstance(img, Image.Image):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            # Handle numpy array (OpenCV)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported image format")
            
            tensors.append(self.trans(img))
            
        return torch.stack(tensors)

    def predict_batch(self, images):
        """
        Predict attributes for a batch of face images.
        """
        if not images:
            return []
            
        input_tensor = self.preprocess_batch(images)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor.to(self.device))
            outputs = outputs.cpu().numpy()
            
        # Process batch outputs
        results = []
        inference_time_batch = (time.time() - start_time) * 1000
        avg_time = inference_time_batch / len(images)
        
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        for output in outputs:
            race_outputs = output[0:7]
            gender_outputs = output[7:9]
            age_outputs = output[9:18]
            
            race_idx = np.argmax(race_outputs)
            gender_idx = np.argmax(gender_outputs)
            age_idx = np.argmax(age_outputs)
            
            race_conf = float(softmax(race_outputs)[race_idx])
            gender_conf = float(softmax(gender_outputs)[gender_idx])
            age_conf = float(softmax(age_outputs)[age_idx])
            
            results.append({
                "race": self.race_labels[race_idx],
                "race_confidence": race_conf,
                "gender": self.gender_labels[gender_idx],
                "gender_confidence": gender_conf,
                "age": self.age_labels[age_idx],
                "age_confidence": age_conf,
                "inference_time_ms": avg_time
            })
            
        return results

    def predict(self, image):
        """
        Predict attributes for a single face image.
        """
        # Re-use predict_batch for single image to maintain consistency
        return self.predict_batch([image])[0]

if __name__ == "__main__":
    # Test
    MODEL_PATH = "models/res34_fair_align_multi_7_20190809.pt"
    if __name__ == "__main__":
        import os
        if os.path.exists(MODEL_PATH):
            predictor = FairFacePredictor(MODEL_PATH)
            # Use dummy image if file not found
            try:
                img = Image.new('RGB', (224, 224), color = 'red')
                result = predictor.predict(img)
                print(result)
            except Exception as e:
                print(e)
