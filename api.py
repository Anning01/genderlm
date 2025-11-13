# -*- coding: utf-8 -*-
import io
import os
import uuid
from collections import deque
from datetime import datetime
from threading import Lock
from typing import List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification

from face_detector import FaceDetector

app = FastAPI(
    title="Gender Classification API",
    description="使用 Vision Transformer 模型进行性别识别，支持人脸检测和批量处理",
    version="2.0.0"
)

# 模型路径配置
LOCAL_MODEL_PATH = "./gender-classification-2"
HF_MODEL_NAME = "rizvandwiki/gender-classification-2"

# 优先使用本地模型
if os.path.exists(LOCAL_MODEL_PATH):
    print(f"✅ 使用本地模型: {LOCAL_MODEL_PATH}")
    model_path = LOCAL_MODEL_PATH
else:
    print(f"⬇️  本地模型不存在，从 Hugging Face 下载: {HF_MODEL_NAME}")
    model_path = HF_MODEL_NAME

# 加载性别识别模型
print("🔄 正在加载性别识别模型...")
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
print("✅ 性别识别模型加载完成!")

# 加载人脸检测模型
print("🔄 正在加载人脸检测模型...")
try:
    # 使用更低的阈值 (0.4) 提高检测灵敏度
    face_detector = FaceDetector(det_thresh=0.4)
    print("✅ 人脸检测模型加载完成!")
    FACE_DETECTION_AVAILABLE = True
except Exception as e:
    print(f"⚠️  人脸检测模型加载失败: {e}")
    print("⚠️  将不支持人脸检测功能")
    face_detector = None
    FACE_DETECTION_AVAILABLE = False


# 队列系统
class TaskQueue:
    """任务队列管理器"""
    def __init__(self, max_size: int = 1000):
        self.tasks = {}  # task_id -> task_info
        self.pending = deque()  # 待处理队列
        self.processing = {}  # 正在处理的任务
        self.completed = {}  # 已完成的任务 (最多保留100个)
        self.failed = {}  # 失败的任务 (最多保留100个)
        self.max_size = max_size
        self.max_history = 100
        self.lock = Lock()

    def add_task(self, task_id: str, files_count: int, use_face_detection: bool = False):
        """添加任务到队列"""
        with self.lock:
            if len(self.tasks) >= self.max_size:
                raise HTTPException(status_code=429, detail="队列已满，请稍后重试")

            task_info = {
                "task_id": task_id,
                "status": "pending",
                "files_count": files_count,
                "processed_count": 0,
                "results": [],
                "use_face_detection": use_face_detection,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None
            }
            self.tasks[task_id] = task_info
            self.pending.append(task_id)
            return task_info

    def start_task(self, task_id: str):
        """开始处理任务"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "processing"
                self.tasks[task_id]["started_at"] = datetime.now().isoformat()
                self.processing[task_id] = self.tasks[task_id]
                if task_id in self.pending:
                    self.pending.remove(task_id)

    def update_progress(self, task_id: str, result: dict):
        """更新任务进度"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["results"].append(result)
                self.tasks[task_id]["processed_count"] += 1

    def complete_task(self, task_id: str):
        """完成任务"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
                self.completed[task_id] = self.tasks[task_id]
                if task_id in self.processing:
                    del self.processing[task_id]

                # 保留最近的历史记录
                if len(self.completed) > self.max_history:
                    oldest = list(self.completed.keys())[0]
                    del self.completed[oldest]
                    del self.tasks[oldest]

    def fail_task(self, task_id: str, error: str):
        """任务失败"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = error
                self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
                self.failed[task_id] = self.tasks[task_id]
                if task_id in self.processing:
                    del self.processing[task_id]

                # 保留最近的历史记录
                if len(self.failed) > self.max_history:
                    oldest = list(self.failed.keys())[0]
                    del self.failed[oldest]
                    del self.tasks[oldest]

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务信息"""
        with self.lock:
            return self.tasks.get(task_id)

    def get_stats(self) -> dict:
        """获取队列统计信息"""
        with self.lock:
            return {
                "total_tasks": len(self.tasks),
                "pending": len(self.pending),
                "processing": len(self.processing),
                "completed": len(self.completed),
                "failed": len(self.failed),
                "queue_capacity": self.max_size
            }


# 全局队列实例
task_queue = TaskQueue()


def predict_single_image(
    image: Image.Image,
    use_face_detection: bool = False,
    face_scale: float = 1.2
) -> dict:
    """
    预测单张图片的性别

    Args:
        image: PIL Image 对象
        use_face_detection: 是否使用人脸检测
        face_scale: 人脸裁剪的缩放比例

    Returns:
        预测结果字典
    """
    result = {
        "face_detected": False,
        "face_crop_applied": False,
        "original_size": image.size
    }

    # 人脸检测和裁剪
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        cropped_face = face_detector.detect_and_crop(image, use_bbox=True, scale=face_scale)
        if cropped_face:
            result["face_detected"] = True
            result["face_crop_applied"] = True
            result["cropped_size"] = cropped_face.size
            image = cropped_face
        else:
            result["face_detected"] = False
            result["warning"] = "未检测到人脸，使用原始图片"

    # 预处理
    inputs = processor(images=image, return_tensors="pt")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()

    # 获取预测结果
    label = model.config.id2label[predicted_class_id]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    confidence = probabilities[predicted_class_id].item()

    # 所有类别的概率
    all_probabilities = {
        model.config.id2label[i]: float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    result.update({
        "gender": label,
        "confidence": round(confidence, 4),
        "probabilities": all_probabilities
    })

    return result


async def process_batch_task(task_id: str, files: List[UploadFile], use_face_detection: bool):
    """后台处理批量任务"""
    try:
        task_queue.start_task(task_id)

        for idx, file in enumerate(files):
            try:
                # 读取图片
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")

                # 预测
                result = predict_single_image(image, use_face_detection)
                result["filename"] = file.filename
                result["index"] = idx

                # 更新进度
                task_queue.update_progress(task_id, result)

            except Exception as e:
                # 单个文件处理失败，记录错误但继续处理其他文件
                error_result = {
                    "filename": file.filename,
                    "index": idx,
                    "error": str(e),
                    "success": False
                }
                task_queue.update_progress(task_id, error_result)

        # 完成任务
        task_queue.complete_task(task_id)

    except Exception as e:
        task_queue.fail_task(task_id, str(e))


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "Gender Classification API v2.0",
        "features": {
            "face_detection": FACE_DETECTION_AVAILABLE,
            "batch_processing": True,
            "queue_system": True
        },
        "endpoints": {
            "/predict": "POST - 单张图片性别识别",
            "/predict/batch": "POST - 批量图片性别识别",
            "/task/{task_id}": "GET - 查询任务状态",
            "/queue/stats": "GET - 队列统计信息",
            "/health": "GET - 健康检查",
            "/docs": "GET - API 文档"
        }
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    stats = task_queue.get_stats()
    return {
        "status": "healthy",
        "gender_model": "loaded",
        "face_detection": "available" if FACE_DETECTION_AVAILABLE else "unavailable",
        "queue_stats": stats
    }


@app.get("/queue/stats")
async def queue_stats():
    """获取队列统计信息"""
    return task_queue.get_stats()


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.post("/predict")
async def predict_gender(
    file: UploadFile = File(...),
    use_face_detection: bool = False,
    face_scale: float = 1.2
):
    """
    单张图片性别识别

    参数:
        file: 上传的图片文件（支持 JPG, PNG 等格式）
        use_face_detection: 是否使用人脸检测和裁剪 (默认: False)
        face_scale: 人脸裁剪的缩放比例 (默认: 1.2)

    返回:
        JSON 格式的预测结果
    """
    try:
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")

        # 检查人脸检测是否可用
        if use_face_detection and not FACE_DETECTION_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="人脸检测功能不可用，请设置 use_face_detection=False"
            )

        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 预测
        result = predict_single_image(image, use_face_detection, face_scale)
        result["filename"] = file.filename
        result["success"] = True

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    use_face_detection: bool = False
):
    """
    批量图片性别识别

    参数:
        files: 上传的图片文件列表
        use_face_detection: 是否使用人脸检测和裁剪 (默认: False)

    返回:
        任务ID，可用于查询处理进度
    """
    try:
        # 检查文件数量
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="至少需要上传一个文件")

        if len(files) > 100:
            raise HTTPException(status_code=400, detail="单次最多上传100个文件")

        # 检查人脸检测是否可用
        if use_face_detection and not FACE_DETECTION_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="人脸检测功能不可用，请设置 use_face_detection=False"
            )

        # 创建任务
        task_id = str(uuid.uuid4())
        task_queue.add_task(task_id, len(files), use_face_detection)

        # 添加后台任务
        background_tasks.add_task(process_batch_task, task_id, files, use_face_detection)

        return JSONResponse(content={
            "success": True,
            "task_id": task_id,
            "files_count": len(files),
            "message": "任务已创建，正在后台处理",
            "query_url": f"/task/{task_id}"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
