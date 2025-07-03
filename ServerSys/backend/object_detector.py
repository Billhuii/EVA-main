import os
import logging
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽 TensorFlow 的低级别日志输出（减少冗余控制台信息）

"""
Detector 类封装了基于 TensorFlow 的目标检测模型加载与推理流程。
支持提取最终检测结果（RCNN输出）与中间结果（RPN区域提议）。
"""
class Detector:
    # 类别映射（将多个原始类别索引归入更高层类别）
    classes = {
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }
    # RPN区域置信度阈值
    rpn_threshold = 0.5
    """
    初始化检测器：加载冻结模型（frozen graph）并准备 TensorFlow Session。
    """
    def __init__(self, model_path='frozen_inference_graph.pb'):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # TensorFlow 配置：GPU按需分配内存
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.model_path = model_path
        self.d_graph = tf.Graph() # 自定义图加载模型

        # 读取并解析 frozen graph 模型文件
        with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
            od_graph_def = tf.compat.v1.GraphDef()
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
        # 将模型导入图中
        with self.d_graph.as_default() as graph:
            tf.import_graph_def(od_graph_def, name='')
        # 使用图创建 Session，准备推理
        self.session = tf.compat.v1.Session(config=config, graph=graph)
        self.logger.info("Object detector initialized")

    """
    执行单张图像推理（RCNN输出 + RPN中间结果提取）。内部使用
    返回包含全部信息的 output_dict。
    """
    def run_inference_for_single_image(self, image, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors  获取当前图中所有张量的名称
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}

            # FOR RCNN final layer results: RCNN 最终检测结果张量映射
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            # FOR RPN intermedia results RPN 中间结果张量映射（手动指定图中节点名）
            key_tensor_map = {
                "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
                                          "/map/while/"
                                          "MultiClassNonMaxSuppression/"
                                          "Gather/Gather:0"),
                "RPN_score": ("BatchMultiClassNonMaxSuppression/"
                              "map/while/"
                              "MultiClassNonMaxSuppression"
                              "/Gather/Gather_2:0"),
                "Resized_shape": ("Preprocessor/map/while"
                                  "/ResizeToRange/stack_1:0"),
            }
            # 将 RPN 中间结果张量添加到 tensor_dict 中
            for key, tensor_name in key_tensor_map.items():
                if tensor_name in all_tensor_names:
                    tensor_dict[tensor_name] = (
                        tf.compat.v1.get_default_graph()
                        .get_tensor_by_name(tensor_name))

            # 输入图像张量
            image_tensor = (tf.compat.v1.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            # Run inference 执行前向推理
            feed_dict = {image_tensor: np.expand_dims(image, 0)}
            output_dict = self.session.run(tensor_dict,
                                           feed_dict=feed_dict)

            # FOR RPN intermedia results 对 RPN 结果进行归一化处理
            w = output_dict[key_tensor_map['Resized_shape']][1]
            h = output_dict[key_tensor_map['Resized_shape']][0]
            input_shape_array = np.array([h, w, h, w])
            output_dict['RPN_box_normalized'] = output_dict[key_tensor_map[
                'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]
            output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]

            # FOR RCNN final layer results
            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            # 转换 RCNN 输出类型
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'][0])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'][0])
        return output_dict

    # 对外推理方法infer
    """
    目标检测主入口：输入 RGB 格式图像，输出检测结果（类别、置信度、框坐标）。

    返回：
        results      : RCNN 最终分类结果（过滤类别）
        results_rpn  : RPN 提议区域（未分类）
    """
    def infer(self, image_np):
        # this output_dict contains both final layer results and RPN results
        # 获取检测输出字典（含 RCNN + RPN）
        output_dict = self.run_inference_for_single_image(image_np, self.d_graph)

        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        # RCNN 推理结果（包含置信度、归一化坐标）
        results = []
        for i in range(len(output_dict['detection_boxes'])):
            object_class = output_dict['detection_classes'][i]
            relevant_class = False
            for k in Detector.classes.keys():
                if object_class in Detector.classes[k]:
                    object_class = k
                    relevant_class = True
                    break
            if not relevant_class:
                continue

            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            confidence = output_dict['detection_scores'][i]
            box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
            results.append((object_class, confidence, box_tuple))

        # Get RPN regions along with classification results
        # rpn results array will have (class, (xmin, xmax, ymin, ymax)) typles
        # RPN 中间结果：作为 region proposals，直接按阈值筛选
        results_rpn = []
        for idx_region, region in enumerate(output_dict['RPN_box_normalized']):
            x = region[1]
            y = region[0]
            w = region[3] - region[1]
            h = region[2] - region[0]
            conf = output_dict['RPN_score'][idx_region]
            # 过滤掉低置信度、面积过小或过大的框
            if conf < Detector.rpn_threshold or w * h == 0.0 or w * h > 0.04:
                continue
            results_rpn.append(("object", conf, (x, y, w, h)))

        return results, results_rpn
