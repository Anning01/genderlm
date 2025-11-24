"""Face detection and cropping utilities."""
import numpy as np
import threading
from PIL import Image
from insightface.app import FaceAnalysis
from typing import Optional, Tuple, List


class FaceDetector:
    """Face detector using InsightFace."""

    def __init__(self, model_name: str = 'buffalo_s', providers: Optional[list] = None, det_size: tuple = (224, 224)):
        """Initialize face detector.

        Args:
            model_name: InsightFace model name (default: buffalo_s)
            providers: ONNX runtime providers (default: auto-detect CUDA)
            det_size: Detection size (default: (224, 224), smaller = faster but may miss faces)
        """
        self.model_lock = threading.Lock()
        
        # Auto-detect CUDA support if providers not specified
        if providers is None:
            try:
                import torch
                if torch.cuda.is_available():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print("🚀 FaceDetector: Using CUDA for face detection")
                else:
                    providers = ['CPUExecutionProvider']
                    print("⚠️  FaceDetector: CUDA not available, using CPU")
            except ImportError:
                providers = ['CPUExecutionProvider']
                print("⚠️  FaceDetector: PyTorch not found, using CPU")

        self.model = FaceAnalysis(name=model_name, providers=providers)
        # Use smaller det_size for better detection (same as 性别检测.py)
        # ctx_id=0 for GPU, ctx_id=-1 for CPU
        try:
            self.model.prepare(ctx_id=0, det_size=det_size)
            print(f"✓ FaceDetector initialized with det_size={det_size}, ctx_id=0 (GPU)")
        except Exception as e:
            print(f"⚠️  GPU initialization failed ({e}), falling back to CPU")
            self.model.prepare(ctx_id=-1, det_size=det_size)
            print(f"✓ FaceDetector initialized with det_size={det_size}, ctx_id=-1 (CPU)")

    def detect(self, image: Image.Image, max_num: int = 1):
        """Detect faces in image.

        Args:
            image: PIL Image
            max_num: Maximum number of faces to detect (0 = unlimited)

        Returns:
            List of face objects with bbox and landmarks
        """
        # Ensure image is in RGB mode (handle P, L, RGBA modes)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL Image (RGB) to numpy array
        img_array = np.array(image)

        # InsightFace expects BGR format (OpenCV convention)
        # Convert RGB to BGR
        img_array = img_array[:, :, ::-1].copy()  # RGB -> BGR, copy to ensure contiguous

        with self.model_lock:
            faces = self.model.get(img_array, max_num=max_num)

        # Free numpy array memory
        del img_array

        return faces

    @staticmethod
    def crop_face_square(image: Image.Image, bbox: np.ndarray, scale: float = 1.2) -> Image.Image:
        """Crop square face region centered on bbox.

        Args:
            image: PIL Image
            bbox: Face bounding box [x1, y1, x2, y2]
            scale: Scale factor for crop area (default: 1.2)

        Returns:
            Cropped PIL Image
        """
        w, h = image.size
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Calculate center and size
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Use longer side as square size
        side_len = int(max(box_w, box_h) * scale)

        # Calculate new boundaries
        new_x1 = max(center_x - side_len // 2, 0)
        new_y1 = max(center_y - side_len // 2, 0)
        new_x2 = min(new_x1 + side_len, w)
        new_y2 = min(new_y1 + side_len, h)

        # Adjust if hit boundary
        new_x1 = max(new_x2 - side_len, 0)
        new_y1 = max(new_y2 - side_len, 0)

        return image.crop((new_x1, new_y1, new_x2, new_y2))

    @staticmethod
    def crop_face_with_landmarks(image: Image.Image, face, scale: float = 1.2) -> Image.Image:
        """Crop face region centered on facial landmarks.

        Args:
            image: PIL Image
            face: Face object with bbox and optional kps (landmarks)
            scale: Scale factor for crop area (default: 1.2)

        Returns:
            Cropped PIL Image
        """
        w, h = image.size
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1
        side_len = int(max(box_w, box_h) * scale)

        # Use landmarks center if available, else bbox center
        if hasattr(face, 'kps') and face.kps is not None:
            center_x = int(face.kps[:, 0].mean())
            center_y = int(face.kps[:, 1].mean())
        else:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

        # Calculate crop region
        new_x1 = max(center_x - side_len // 2, 0)
        new_y1 = max(center_y - side_len // 2, 0)
        new_x2 = min(new_x1 + side_len, w)
        new_y2 = min(new_y1 + side_len, h)

        # Adjust if hit boundary
        new_x1 = max(new_x2 - side_len, 0)
        new_y1 = max(new_y2 - side_len, 0)

        return image.crop((new_x1, new_y1, new_x2, new_y2))

    def detect_and_crop(
        self,
        image: Image.Image,
        use_bbox: bool = True,
        scale: float = 1.2
    ) -> Optional[Image.Image]:
        """Detect face and return cropped image.

        Args:
            image: PIL Image
            use_bbox: Use bbox-based cropping if True, else use landmark-based
            scale: Scale factor for crop area

        Returns:
            Cropped face image, or None if no face detected
        """
        faces = self.detect(image, max_num=1)
        if not faces:
            return None
        face = faces[0]
        if use_bbox:
            return self.crop_face_square(image, face.bbox, scale)
        else:
            return self.crop_face_with_landmarks(image, face, scale)

    def detect_and_crop_all(
        self,
        image: Image.Image,
        use_bbox: bool = True,
        scale: float = 1.2,
        max_num: int = 0
    ) -> Tuple[List[Image.Image], List]:
        """Detect all faces and return list of cropped images.

        Args:
            image: PIL Image
            use_bbox: Use bbox-based cropping if True, else use landmark-based
            scale: Scale factor for crop area
            max_num: Maximum number of faces to detect (0 = unlimited)

        Returns:
            Tuple of (cropped_images_list, faces_info_list)
            - cropped_images_list: List of cropped PIL Images
            - faces_info_list: List of face objects with bbox and landmarks
            Returns ([], []) if no faces detected
        """
        faces = self.detect(image, max_num=max_num)
        if not faces:
            return [], []

        cropped_images = []
        for face in faces:
            if use_bbox:
                cropped = self.crop_face_square(image, face.bbox, scale)
            else:
                cropped = self.crop_face_with_landmarks(image, face, scale)
            cropped_images.append(cropped)

        return cropped_images, faces



if __name__ == '__main__':
    import os
    import time

    print("=" * 60)
    print("Face Detector Test")
    print("=" * 60)

    # Initialize detector
    print("\n1. 初始化人脸检测器...")
    start_time = time.time()
    detector = FaceDetector()
    init_time = time.time() - start_time
    print(f"   ✓ 人脸检测器加载完成 (耗时: {init_time:.3f}秒)")
    
    # Test images
    test_images = [
        # "example/avatar_f.png",
        # "example/avatar_w.png",
        # "example/avatar_team.png",
        "example/img.png"
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\n⚠ 警告: {img_path} 不存在, 跳过中...")
            continue

        img_start_time = time.time()
        print(f"\n2. 处理图片: {img_path}")
        print("-" * 60)

        # Load image
        load_start = time.time()
        image = Image.open(img_path)
        load_time = time.time() - load_start
        print(f"   图像大小: {image.size} (加载耗时: {load_time:.5f}秒)")

        # Detect faces
        detect_start = time.time()
        faces = detector.detect(image, max_num=0)  # Detect all faces
        detect_time = time.time() - detect_start
        print(f"   检测到的人脸: {len(faces)} (检测耗时: {detect_time:.5f}秒)")
        
        if len(faces) == 0:
            print("   ✗ 未检测到人脸")
            continue
        
        # Display face info
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            print(f"\n   Face {i+1}:")
            print(f"     - Bounding box: {bbox}")
            if hasattr(face, 'kps') and face.kps is not None:
                print(f"     - Landmarks: {face.kps.shape[0]} points")
                landmark_center = (int(face.kps[:, 0].mean()), int(face.kps[:, 1].mean()))
                print(f"     - Landmark center: {landmark_center}")
        
        # Test cropping methods
        face = faces[0]  # Use first face

        # Method 1: Bbox-based crop
        print("\n   测试基于bbox的裁剪...")
        crop_start = time.time()
        cropped_bbox = detector.crop_face_square(image, face.bbox, scale=1.2)
        crop_time = time.time() - crop_start
        output_bbox = img_path.replace(".png", "_crop_bbox.png")
        cropped_bbox.save(output_bbox)
        print(f"     ✓ 保存至: {output_bbox} (耗时: {crop_time:.5f}秒)")
        print(f"     大小: {cropped_bbox.size}")

        # Method 2: Landmark-based crop
        print("\n   测试基于Landmark的裁剪...")
        crop_start = time.time()
        cropped_landmark = detector.crop_face_with_landmarks(image, face, scale=1.2)
        crop_time = time.time() - crop_start
        output_landmark = img_path.replace(".png", "_crop_landmark.png")
        cropped_landmark.save(output_landmark)
        print(f"     ✓ 保存至: {output_landmark} (耗时: {crop_time:.5f}秒)")
        print(f"     大小: {cropped_landmark.size}")

        # Method 3: detect_and_crop convenience method
        print("\n   测试detect_and_crop方法...")
        crop_start = time.time()
        cropped_auto = detector.detect_and_crop(image, use_bbox=True, scale=1.5)
        crop_time = time.time() - crop_start
        if cropped_auto:
            output_auto = img_path.replace(".png", "_crop_auto.png")
            cropped_auto.save(output_auto)
            print(f"     ✓ 保存至: {output_auto} (耗时: {crop_time:.5f}秒)")
            print(f"     大小: {cropped_auto.size}")

        img_total_time = time.time() - img_start_time
        print(f"\n   >>> 该图片总耗时: {img_total_time:.5f}秒 <<<")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)
