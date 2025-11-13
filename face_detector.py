"""Face detection and cropping utilities."""
from typing import Optional

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class FaceDetector:
    """Face detector using InsightFace."""

    def __init__(
        self,
        model_name: str = 'buffalo_s',
        providers: Optional[list] = None,
        det_size: tuple = (640, 640),
        det_thresh: float = 0.5
    ):
        """Initialize face detector.

        Args:
            model_name: InsightFace model name (default: buffalo_s)
            providers: ONNX runtime providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            det_size: Detection size (default: (640, 640))
            det_thresh: Detection threshold, lower = more sensitive (default: 0.5, range: 0.0-1.0)
        """
        self.model = FaceAnalysis(name=model_name, providers=providers)
        self.model.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def detect(self, image: Image.Image, max_num: int = 1):
        """Detect faces in image.

        Args:
            image: PIL Image (RGB format)
            max_num: Maximum number of faces to detect (0 = unlimited)

        Returns:
            List of face objects with bbox and landmarks
        """
        # Convert PIL Image (RGB) to numpy array
        img_array = np.array(image)

        # InsightFace expects BGR format (OpenCV convention)
        # Convert RGB to BGR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array[:, :, ::-1]  # RGB -> BGR

        faces = self.model.get(img_array, max_num=max_num)
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



if __name__ == '__main__':
    import os
    import time

    print("=" * 60)
    print("Face Detector Test")
    print("=" * 60)

    # Test images
    test_images = [
        "gender-classification-2/images/male.jpg",
    ]

    # Test with different detection thresholds
    thresholds = [0.5, 0.3]  # Default and more sensitive

    for thresh in thresholds:
        print(f"\n{'=' * 60}")
        print(f"Testing with detection threshold: {thresh}")
        print(f"{'=' * 60}")

        # Initialize detector
        print(f"\n1. 初始化人脸检测器 (阈值={thresh})...")
        start_time = time.time()
        detector = FaceDetector(det_thresh=thresh)
        init_time = time.time() - start_time
        print(f"   ✓ 人脸检测器加载完成 (耗时: {init_time:.3f}秒)")

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
                print(f"   ✗ 未检测到人脸 (阈值={thresh})")
                continue

            # Display face info
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                print(f"\n   Face {i+1}:")
                print(f"     - Bounding box: {bbox}")
                print(f"     - Detection score: {face.det_score:.4f}")
                if hasattr(face, 'kps') and face.kps is not None:
                    print(f"     - Landmarks: {face.kps.shape[0]} points")
                    landmark_center = (int(face.kps[:, 0].mean()), int(face.kps[:, 1].mean()))
                    print(f"     - Landmark center: {landmark_center}")

            # Test cropping only for first threshold to avoid duplicate outputs
            if thresh == thresholds[0] and len(faces) > 0:
                face = faces[0]  # Use first face

                # Get base filename without extension
                base_path = os.path.splitext(img_path)[0]
                ext = os.path.splitext(img_path)[1]

                # Method 1: Bbox-based crop
                print("\n   测试基于bbox的裁剪...")
                crop_start = time.time()
                cropped_bbox = detector.crop_face_square(image, face.bbox, scale=1.2)
                crop_time = time.time() - crop_start
                output_bbox = f"{base_path}_crop_bbox{ext}"
                cropped_bbox.save(output_bbox)
                print(f"     ✓ 保存至: {output_bbox} (耗时: {crop_time:.5f}秒)")
                print(f"     大小: {cropped_bbox.size}")

                # Method 2: Landmark-based crop
                print("\n   测试基于Landmark的裁剪...")
                crop_start = time.time()
                cropped_landmark = detector.crop_face_with_landmarks(image, face, scale=1.2)
                crop_time = time.time() - crop_start
                output_landmark = f"{base_path}_crop_landmark{ext}"
                cropped_landmark.save(output_landmark)
                print(f"     ✓ 保存至: {output_landmark} (耗时: {crop_time:.5f}秒)")
                print(f"     大小: {cropped_landmark.size}")

                # Method 3: detect_and_crop convenience method
                print("\n   测试detect_and_crop方法...")
                crop_start = time.time()
                cropped_auto = detector.detect_and_crop(image, use_bbox=True, scale=1.5)
                crop_time = time.time() - crop_start
                if cropped_auto:
                    output_auto = f"{base_path}_crop_auto{ext}"
                    cropped_auto.save(output_auto)
                    print(f"     ✓ 保存至: {output_auto} (耗时: {crop_time:.5f}秒)")
                    print(f"     大小: {cropped_auto.size}")

            img_total_time = time.time() - img_start_time
            print(f"\n   >>> 该图片总耗时: {img_total_time:.5f}秒 <<<")

    print("\n" + "=" * 60)
    print("✓ 测试完成!")
    print("\n💡 建议:")
    print("  - 如果某些图片检测不到人脸，尝试降低 det_thresh (0.3 或更低)")
    print("  - 如果检测到太多误报，尝试提高 det_thresh (0.6 或更高)")
    print("  - 默认推荐值: 0.5")
    print("=" * 60)
