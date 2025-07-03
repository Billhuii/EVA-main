import os
import logging
from .object_detector import Detector
from ..utils import (Results, Region)
import cv2 as cv

"""
    Server 类封装了目标检测推理接口，用于处理单张图像，返回检测框结果。
"""
class Server:
    def __init__(self):
        # 设置服务器日志记录器（但不主动输出）
        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        # 初始化目标检测器（Detector类通常封装了模型加载与推理）
        self.detector = Detector()
        self.logger.info("Server started")

    """
        对指定图像文件执行目标检测。

        参数:
            images_direc (str): 图像所在目录
            resolution (float): 图像分辨率标识（可用于区分多源图像）
            fname (str): 文件名（如"0000000001.png"）

        返回:
            (final_results, rpn_regions):
            - final_results: 检测结果封装为 Results 对象
            - rpn_regions: RPN 提议区域的封装（供后续合并等处理）
    """
    def perform_detection(self, images_direc, resolution, fname=None):
        final_results = Results()  # 存放最终检测结果（供合并后使用）
        rpn_regions = Results()  # 存放原始 RPN 结果（可用于对比分析）
        # read image ；解析帧号 fid（从文件名中提取，例如 "0000000001.png" -> 1）
        fid = int(fname.split(".")[0])
        # 加载图像并转为 RGB 格式
        image_path = os.path.join(images_direc, fname)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # detect image
        # 执行推理（返回两个列表：检测结果 + RPN 结果）
        detection_results, rpn_results = (
            self.detector.infer(image))
        frame_with_no_results = True
        # 遍历检测结果，构造 Region 对象，加入 final_results
        for label, conf, (x, y, w, h) in detection_results:
            if w * h == 0.0:
                continue # 忽略无效框
            r = Region(fid, x, y, w, h, conf, label,
                       resolution)
            final_results.append(r)
        # 遍历 RPN 原始结果，构造 Region 对象，加入 rpn_regions
        for label, conf, (x, y, w, h) in rpn_results:
            r = Region(fid, x, y, w, h, conf, label,
                       resolution)
            rpn_regions.append(r)
        # 记录日志（若启用日志级别可输出推理数量）
        self.logger.debug(
            f"Got {len(final_results)} results "
            f"and {len(rpn_regions)} for {fname}")

        return final_results, rpn_regions

