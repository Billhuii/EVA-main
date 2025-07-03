# ----------- GPU 配置部分：仅用于避免显存一次性全部占用 -----------

import tensorflow as tf

# 列出所有物理 GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# 如果检测到 GPU，设置按需分配显存（防止 OOM）
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ----------- 模块导入 -----------

import core.utils as yolo_utils  # 自定义后处理函数
from tensorflow.python.saved_model import tag_constants  # TensorFlow 模型标签
import cv2  # OpenCV 图像处理
import numpy as np
from tensorflow.compat.v1 import ConfigProto  # 为了兼容 TF1 风格配置 GPU

# ----------- 加载预训练 YOLOv4 模型 -----------

# 加载已保存的 YOLOv4 模型（在 ./checkpoints-yolo/yolov4-416 路径下）
# 该模型使用的是 TensorFlow SavedModel 格式
saved_model_loaded = tf.saved_model.load(
    "./checkpoints-yolo/yolov4-416", tags=[tag_constants.SERVING]
)

# ----------- 物体检测主函数 -----------

def detect(img_dir):
    """
    输入：
        img_dir: 图像路径
    输出：
        返回 yolo_utils.targets() 提取出的目标结构列表
    """

    # 配置 GPU 资源限制（与顶部相同目的，兼容性配置）
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    input_size = 416  # 模型输入尺寸固定为 416×416
    image_path = img_dir

    # ---------- 图像预处理 ----------
    # 读取原始图像（BGR）
    original_image = cv2.imread(image_path)
    # 转换为 RGB 格式，符合模型输入要求
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 调整尺寸为 416×416 并归一化到 0~1
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    # 封装为 batch（batch size = 1）
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # ---------- 推理 ----------
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)  # 调用模型进行前向传播

    # 获取预测输出字典中的内容（boxes + conf）
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]         # 边界框坐标
        pred_conf = value[:, :, 4:]      # 置信度 + 类别概率

    # ---------- 后处理 ----------
    # NMS 非极大值抑制，去除重叠框
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,      # IoU 阈值
        score_threshold=0.25     # 置信度阈值
    )

    # 将张量转为 NumPy 数组
    pred_bbox = [
        boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()
    ]

    # 调用 yolo_utils.targets 提取目标结构
    return yolo_utils.targets(original_image, pred_bbox)



# import tensorflow as tf
#
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import core.utils as yolo_utils
# from tensorflow.python.saved_model import tag_constants
# import cv2
# import numpy as np
# from tensorflow.compat.v1 import ConfigProto
#
# saved_model_loaded = tf.saved_model.load("./checkpoints-yolo/yolov4-416", tags=[tag_constants.SERVING])
#
#
# def detect(img_dir):
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     input_size = 416
#     image_path = img_dir
#
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#
#     image_data = cv2.resize(original_image, (input_size, input_size))
#     image_data = image_data / 255.
#
#     images_data = []
#     for i in range(1):
#         images_data.append(image_data)
#     images_data = np.asarray(images_data).astype(np.float32)
#     infer = saved_model_loaded.signatures['serving_default']
#     batch_data = tf.constant(images_data)
#     pred_bbox = infer(batch_data)
#     for key, value in pred_bbox.items():
#         boxes = value[:, :, 0:4]
#         pred_conf = value[:, :, 4:]
#
#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#         boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#         scores=tf.reshape(
#             pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#         max_output_size_per_class=50,
#         max_total_size=50,
#         iou_threshold=0.45,
#         score_threshold=0.25
#     )
#     pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
#     return yolo_utils.targets(original_image, pred_bbox)
