"""
predict low confidence objects
"""
import os
from backend.server import Server
from utils import merge_boxes_in_results

# 初始化目标检测服务器
server = Server()

# 临时图像路径（待重新推理的图像保存在该目录）
image_direc = f"./server_temp/"

# 打开低置信度图像记录文件
f = open("./backend/low_img.txt", "r")
# 打开高置信度检测结果文件（以追加形式写入）
results_file = open("./backend/high_img.txt", "a")

# 对低置信度图像逐个重新推理
for line in f:
    name = line.split(",")[0]  # 图像文件名
    img_path = f"./server_temp/{name}"

    # 检查图像是否存在（防止路径问题）
    if os.path.exists(img_path):
        # 解析原始检测框信息（可用于与原始结果对比）
        y1 = line.split(",")[1]
        x1 = line.split(",")[2]
        y2 = line.split(",")[3]
        x2 = line.split(",")[4]
        conf = line.split(",")[5]
        label = line.split(",")[6]

        # 重新进行完整目标检测
        results, rpn_results = server.perform_detection(image_direc, 1.0, name)

        # 对结果进行后处理（合并重叠框等）
        results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)

        # 将新检测结果写入高置信度结果文件
        for region in results.regions:
            conf = region.conf
            label = region.label
            str_to_write = f"{name}, {y1}, {x1}, {y2}, {x2}, {conf}, {label}\n"
            results_file.write(str_to_write)

f.close()
results_file.close()




# """
# predict low confidence objects
# """
# import os
# from backend.server import Server
# from utils import merge_boxes_in_results
# server = Server()
# image_direc = f"./server_temp/"
# f = open("./backend/low_img.txt", "r")
# results_file = open("./backend/high_img.txt", "a")
# for line in f:
#     name = line.split(",")[0]
#     img_path = f"./server_temp/{name}"
#     if os.path.exists(img_path):
#         y1 = line.split(",")[1]
#         x1 = line.split(",")[2]
#         y2 = line.split(",")[3]
#         x2 = line.split(",")[4]
#         conf = line.split(",")[5]
#         label = line.split(",")[6]
#         #if float(conf) < 0.8:
#         results, rpn_results = server.perform_detection(image_direc, 1.0, name)
#         results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)
#         for region in results.regions:
#             # prepare the string to write
#             conf = region.conf
#             label = region.label
#             str_to_write = f"{name}, {y1}, {x1}, {y2}, {x2}, {conf}, {label}\n"
#             results_file.write(str_to_write)
#         # else:
#         #     results_file.write(line)
# f.close()
# results_file.close()