# -*- coding: utf-8 -*-
import gradio as gr
from PIL import Image
import torch
import os
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
from face_detector import FaceDetector

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


def predict_gender(image, use_face_detection=False, face_scale=1.2):
    """
    性别识别函数

    参数:
        image: PIL Image 对象或 numpy array
        use_face_detection: 是否使用人脸检测
        face_scale: 人脸裁剪缩放比例

    返回:
        (结果字典, 处理后的图片, 信息文本)
    """
    if image is None:
        return {"错误": 1.0}, None, "❌ 请上传图片"

    info_lines = []
    processed_image = image

    try:
        # 确保图片是 RGB 格式
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")

        original_size = image.size
        info_lines.append(f"📐 原始图片尺寸: {original_size[0]} x {original_size[1]}")

        # 人脸检测和裁剪
        face_detected = False
        if use_face_detection and FACE_DETECTION_AVAILABLE:
            info_lines.append("🔍 正在进行人脸检测...")
            cropped_face = face_detector.detect_and_crop(image, use_bbox=True, scale=face_scale)

            if cropped_face:
                face_detected = True
                info_lines.append("✅ 检测到人脸，已自动裁剪")
                info_lines.append(f"✂️  裁剪后尺寸: {cropped_face.size[0]} x {cropped_face.size[1]}")
                image = cropped_face
                processed_image = cropped_face
            else:
                info_lines.append("⚠️  未检测到人脸，使用原始图片")
        elif use_face_detection and not FACE_DETECTION_AVAILABLE:
            info_lines.append("⚠️  人脸检测功能不可用")

        # 预处理
        inputs = processor(images=image, return_tensors="pt")

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 计算概率
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

        # 构建结果字典
        results = {
            model.config.id2label[i]: float(probabilities[i].item())
            for i in range(len(probabilities))
        }

        # 生成详细信息
        predicted_label = max(results.items(), key=lambda x: x[1])
        info_lines.append(f"\n🎯 识别结果: {predicted_label[0]}")
        info_lines.append(f"📊 置信度: {predicted_label[1]:.2%}")

        if face_detected:
            info_lines.append(f"🔧 人脸裁剪缩放: {face_scale}x")

        info_text = "\n".join(info_lines)

        return results, processed_image, info_text

    except Exception as e:
        error_msg = f"❌ 处理图片时出错: {str(e)}"
        return {"错误": 1.0}, None, error_msg


def predict_batch(files, use_face_detection=False, face_scale=1.2):
    """
    批量图片识别

    参数:
        files: 文件列表
        use_face_detection: 是否使用人脸检测
        face_scale: 人脸裁剪缩放比例

    返回:
        结果文本
    """
    if not files or len(files) == 0:
        return "❌ 请上传至少一张图片"

    results_text = [f"📦 批量处理 {len(files)} 张图片\n{'='*50}\n"]

    for idx, file in enumerate(files, 1):
        try:
            # 读取图片
            if isinstance(file, str):
                image = Image.open(file).convert("RGB")
                filename = os.path.basename(file)
            else:
                image = Image.open(file.name).convert("RGB")
                filename = os.path.basename(file.name)

            # 人脸检测
            face_info = ""
            if use_face_detection and FACE_DETECTION_AVAILABLE:
                cropped = face_detector.detect_and_crop(image, use_bbox=True, scale=face_scale)
                if cropped:
                    image = cropped
                    face_info = " [人脸已裁剪]"
                else:
                    face_info = " [未检测到人脸]"

            # 预测
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            predicted_id = logits.argmax(-1).item()
            label = model.config.id2label[predicted_id]
            confidence = probabilities[predicted_id].item()

            # 添加结果
            results_text.append(
                f"{idx}. {filename}{face_info}\n"
                f"   性别: {label} | 置信度: {confidence:.2%}\n"
            )

        except Exception as e:
            results_text.append(f"{idx}. 处理失败: {str(e)}\n")

    results_text.append(f"\n{'='*50}\n✅ 批量处理完成!")
    return "".join(results_text)


# 创建 Gradio 界面
with gr.Blocks(
    title="性别识别系统",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        #title {
            text-align: center;
            color: #2563eb;
        }
        #description {
            text-align: center;
            color: #64748b;
        }
        .info-box {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
    """
) as demo:
    gr.Markdown(
        """
        # 🎭 图像性别识别系统
        基于 Vision Transformer (ViT) 的人物性别识别服务 | 支持人脸智能检测与裁剪
        """,
        elem_id="title"
    )

    face_detection_status = "✅ 人脸检测可用" if FACE_DETECTION_AVAILABLE else "⚠️ 人脸检测不可用"
    gr.Markdown(
        f"""
        系统状态: 性别识别模型已加载 | {face_detection_status}
        """,
        elem_id="description"
    )

    with gr.Tabs():
        # Tab 1: 单图识别
        with gr.TabItem("🖼️ 单图识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="上传图片",
                        type="pil",
                        sources=["upload", "clipboard", "webcam"],
                        height=400
                    )

                    with gr.Row():
                        use_face_det = gr.Checkbox(
                            label="启用人脸检测",
                            value=False,
                            interactive=FACE_DETECTION_AVAILABLE
                        )
                        face_scale_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="人脸裁剪缩放比例",
                            interactive=FACE_DETECTION_AVAILABLE
                        )

                    predict_btn = gr.Button(
                        "🔍 开始识别",
                        variant="primary",
                        size="lg"
                    )

                    clear_btn = gr.Button(
                        "🗑️ 清除",
                        variant="secondary"
                    )

                with gr.Column(scale=1):
                    output_label = gr.Label(
                        label="识别结果",
                        num_top_classes=2,
                        show_label=True
                    )

                    processed_image = gr.Image(
                        label="处理后的图片",
                        type="pil",
                        height=300
                    )

                    info_text = gr.Textbox(
                        label="处理信息",
                        lines=8,
                        max_lines=15
                    )

            gr.Markdown(
                """
                ### 📊 功能说明
                - **基础识别**: 直接对上传的图片进行性别识别
                - **人脸检测**: 自动检测并裁剪人脸区域，提高识别准确度
                - **缩放调节**: 调整人脸裁剪范围（1.0=紧贴人脸，2.0=包含更多背景）

                模型准确率: **99.1%**
                """
            )

        # Tab 2: 批量识别
        with gr.TabItem("📁 批量识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="上传多张图片",
                        file_count="multiple",
                        file_types=["image"]
                    )

                    with gr.Row():
                        batch_use_face = gr.Checkbox(
                            label="启用人脸检测",
                            value=False,
                            interactive=FACE_DETECTION_AVAILABLE
                        )
                        batch_scale = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="人脸裁剪缩放比例",
                            interactive=FACE_DETECTION_AVAILABLE
                        )

                    batch_btn = gr.Button(
                        "🚀 批量识别",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    batch_results = gr.Textbox(
                        label="批量识别结果",
                        lines=20,
                        max_lines=30
                    )

            gr.Markdown(
                """
                ### 📦 批量处理说明
                - 支持同时上传多张图片进行批量识别
                - 可选择是否对每张图片进行人脸检测
                - 结果会显示每张图片的识别结果和置信度
                """
            )

    # 示例图片提示（Gradio 3.x 版本兼容）
    gr.Markdown(
        """
        ---
        ### 📁 示例图片
        如果已下载模型，可以使用以下路径的示例图片进行测试：
        - `gender-classification-2/images/female.jpg`
        - `gender-classification-2/images/male.jpg`

        ---
        💡 **提示**:
        - 为获得最佳效果，请上传清晰的人物照片
        - 启用人脸检测可以提高识别准确度，特别是在包含背景的照片中
        - 批量处理时，建议每次上传不超过50张图片
        """
    )

    # 事件绑定 - 单图识别
    predict_btn.click(
        fn=predict_gender,
        inputs=[input_image, use_face_det, face_scale_slider],
        outputs=[output_label, processed_image, info_text]
    )

    clear_btn.click(
        fn=lambda: (None, None, None, ""),
        inputs=None,
        outputs=[input_image, output_label, processed_image, info_text]
    )

    # 自动预测（可选）
    input_image.change(
        fn=predict_gender,
        inputs=[input_image, use_face_det, face_scale_slider],
        outputs=[output_label, processed_image, info_text]
    )

    # 事件绑定 - 批量识别
    batch_btn.click(
        fn=predict_batch,
        inputs=[batch_files, batch_use_face, batch_scale],
        outputs=batch_results
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
