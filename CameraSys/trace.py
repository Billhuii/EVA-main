import math

# IoU 阈值，超过该值则认为两个目标是同一个目标
THRESHOLD = 0.6

# 计算两个矩形框的交集面积
def calc_intersection_area(a, b):
    to = max(a[1], b[1])  # top
    le = max(a[0], b[0])  # left
    bo = min(a[3], b[3])  # bottom
    ri = min(a[2], b[2])  # right

    w = max(0, ri - le)
    h = max(0, bo - to)
    return w * h

# 计算单个边界框的面积
def calc_area(a):
    w = max(0, a[2] - a[0])
    h = max(0, a[3] - a[1])
    return w * h

# 计算两个边界框的 IoU（交并比）
def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area

# 输入：上一帧和当前帧中的目标列表
# 输出：清洗后的上一帧目标列表、当前帧目标列表、配对结果列表（res）
def preprocess_data(last_frame, current_frame):
    res = []      # 最终配对结果列表
    temp = []     # 暂存被配对的上一帧目标（用于后续重新添加）
    to_remove = []  # 冗余目标集合

    # 步骤一：清洗当前帧中出现的冗余目标（多个框重复检测同一个目标）
    for index, target in enumerate(current_frame):
        for l_index, l_target in enumerate(current_frame):
            if index is not l_index:
                # 获取两个目标的边界框
                a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
                b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
                iou = calc_iou(a, b)

                # 同一帧内 IoU > 阈值 + 来源同一张图片，则视为重复目标，只保留置信度高的
                if iou > THRESHOLD and target['name'].split(".")[0].split("_")[1] == l_target['name'].split(".")[0].split("_")[1]:
                    if target['confidence'] > l_target['confidence']:
                        to_remove.append(l_target)
                    else:
                        to_remove.append(target)

    # 移除当前帧中被判定为冗余的目标
    for item in to_remove:
        if item in current_frame:
            current_frame.remove(item)

    # 步骤二：跨帧匹配当前帧和上一帧中相同目标
    if len(last_frame) > 0:
        for index, target in enumerate(current_frame):
            max_iou = 0           # 当前目标与上一帧所有目标的最大 IoU
            temp_pair = []        # 当前找到的最佳配对目标
            remove_index = 0      # 记录匹配目标在 last_frame 中的索引
            remove_target = []    # 记录匹配目标对象

            a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
            for l_index, l_target in enumerate(last_frame):
                b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
                iou = calc_iou(a, b)

                # 保留 IoU 最大的配对
                if iou > THRESHOLD and iou > max_iou:
                    temp_pair = [l_target['name'], target['name']]
                    max_iou = iou
                    remove_index = l_index
                    remove_target = l_target

            if temp_pair:
                res.append(temp_pair)                      # 保存配对结果
                last_frame.pop(remove_index)               # 从上一帧中移除被匹配的目标
                temp.append(remove_target)                 # 保存以便最后重新合并进 last_frame

    # 把本次匹配中被移除的目标重新合并到上一帧中
    last_frame.extend(temp)

    return last_frame, current_frame, res




# import math
#
# # Trace threshold
# THRESHOLD = 0.6
#
#
# def calc_intersection_area(a, b):
#     to = max(a[1], b[1])
#     le = max(a[0], b[0])
#     bo = min(a[3], b[3])
#     ri = min(a[2], b[2])
#
#     w = max(0, ri - le)
#     h = max(0, bo - to)
#
#     return w * h
#
#
# def calc_area(a):
#     w = max(0, a[2] - a[0])
#     h = max(0, a[3] - a[1])
#
#     return w * h
#
#
# def calc_iou(a, b):
#     intersection_area = calc_intersection_area(a, b)
#     union_area = calc_area(a) + calc_area(b) - intersection_area
#     return intersection_area / union_area
#
#
# # Return target pairs in two adjacent frames that detected to present same objects
# def preprocess_data(last_frame, current_frame):
#     res = []
#     temp = []
#     to_remove = []
#     for index, target in enumerate(current_frame):
#         for l_index, l_target in enumerate(current_frame):
#             if index is not l_index:
#                 a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
#                 b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
#                 iou = calc_iou(a, b)
#                 if iou > THRESHOLD and target['name'].split(".")[0].split("_")[1] == \
#                         l_target['name'].split(".")[0].split("_")[1]:
#                     # print('dump: ', l_target['name'], target['name'])
#                     if target['confidence'] > l_target['confidence']:
#                         to_remove.append(l_target)
#                     else:
#                         to_remove.append(target)
#     # print(to_remove)
#     for item in to_remove:
#         if item in current_frame:
#             current_frame.remove(item)
#
#     if len(last_frame) > 0:
#         for index, target in enumerate(current_frame):
#             max_iou = 0
#             temp_pair = []
#             remove_index = 0
#             remove_target = []
#             a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
#             for l_index, l_target in enumerate(last_frame):
#                 b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
#                 iou = calc_iou(a, b)
#                 if iou > THRESHOLD:
#                     if iou > max_iou:
#                         temp_pair = [l_target['name'], target['name']]
#                         max_iou = iou
#                         remove_index = l_index
#                         remove_target = l_target
#             # print(target['name'], remove_target['name'], max_iou)
#             if temp_pair:
#                 # print(temp_pair)
#                 res.append(temp_pair)
#                 last_frame.pop(remove_index)
#                 temp.append(remove_target)
#     last_frame.extend(temp)
#     return last_frame, current_frame, res
