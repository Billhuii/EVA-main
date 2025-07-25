#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

#cfg 是一个嵌套的 EasyDict 对象（字典结构可用点操作访问字段，如 cfg.YOLO.CLASSES）
cfg                           = __C


# __C.YOLO.CLASSES         # 类别名称文件路径（如 COCO 的80类）
# __C.YOLO.ANCHORS         # 预设 anchor 框，用于目标框回归
# __C.YOLO.STRIDES         # 对应三个尺度输出特征图的下采样倍数
# __C.YOLO.XYSCALE         # 坐标缩放系数，用于预测框位置回归增强
# __C.YOLO.IOU_LOSS_THRESH # IOU 阈值，用于背景预测时的损失计算

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# __C.TRAIN.ANNOT_PATH     # 训练标注文件路径
# __C.TRAIN.BATCH_SIZE     # 训练批大小
# __C.TRAIN.INPUT_SIZE     # 输入图片大小（固定或多尺度）
# __C.TRAIN.DATA_AUG       # 是否使用数据增强
# __C.TRAIN.LR_INIT/END    # 初始和最终学习率
# __C.TRAIN.WARMUP_EPOCHS  # 预热轮数
# __C.TRAIN.FISRT_STAGE_EPOCHS / SECOND_STAGE_EPOCHS # 冻结阶段和解冻阶段的训练轮数

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30


# __C.TEST.ANNOT_PATH          # 测试标注路径
# __C.TEST.BATCH_SIZE          # 测试批大小
# __C.TEST.SCORE_THRESHOLD     # 分数阈值
# __C.TEST.IOU_THRESHOLD       # NMS时的IOU阈值
# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


