import json
import os
import logging
import shutil
from flask import Flask, request, make_response
import flask
from server import Server
from utils import merge_boxes_in_results

# C：当前置信度阈值；
# del_C：阈值增减的步长；
# 其他变量用于动态调整 C
app = Flask(__name__)
D_above, D_below, a_num, b_num = 0, 0, 0, 0
del_C = 0.05
C = 0.9
server = Server()

@app.before_first_request
def init():
    """
    在第一次请求之前运行：
    - 清空或新建 `low_img.txt`, `high_img.txt` 两个结果文件
    - 重建 `server_temp/` 临时目录（用于保存上传的图像）
    """
    global server
    for file in ['./low_img.txt', './high_img.txt']:
        if os.path.isfile(file):
            os.remove(file)
        f = open(file, 'w+')
        f.close()
    for dirs in ['../server_temp']:
        if os.path.isdir(dirs):
            shutil.rmtree(dirs)
        os.mkdir(dirs)


@app.route("/")
@app.route("/index")
def index(): # 首页测试接口
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


# receive low confidence results
# /low 接口：接收低置信度推理结果
@app.route("/low", methods=["POST"])
def perform_low_images():
    global D_above, D_below, a_num, b_num, C, del_C
    change_threshold = 0.0

    result = request.form
    image = request.files["image"]
    image.save(os.path.join('../server_temp/', result['name']))
    # 记录低置信度结果信息（名称、框位置、conf、类别）到 low_img.txt
    result_file = open("low_img.txt", "a")
    result_file.write(f"{result.get('name')}, {result.getlist('shape')[0]}, {result.getlist('shape')[1]},"
                      f"{result.getlist('shape')[2]}, {result.getlist('shape')[3]}, {result.get('conf')}, {result.get('label')}\n")
    result_file.close()

    """
    The code to run Alg.2 
    """
    # 如果置信度 ≥ C，则 D_above 统计 +1
    # 否则（且不远离 C），D_below 统计 +1
    # 如果检测结果 label 和高置信度区域匹配，也分别统计 a_num、b_num
    # 如果高置信误判率高 → 增加阈值；低置信但预测准确率高 → 降低阈值

    # predict low
    # conf = float(result.get('conf'))
    # if conf >= C:
    #     D_above += 1
    # else:
    #     if C - conf < 0.2:
    #         D_below += 1
    # label = result.get('label')
    # results, rpn_results = server.perform_detection(f"../server_temp/", 1.0, result.get('name'))
    # results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)
    # for region in results.regions:
    #     if region.label is "vehicle" and label in ["car", "truck", "bicycle", "motorbike"]:
    #         if conf >= C:
    #             a_num += 1
    #         else:
    #             if C - conf < 0.2:
    #                 b_num += 1
    # print(a_num, D_above, b_num, D_below)
    # if a_num / D_above <= 0.9:
    #     change_threshold += 3 * del_C
    # # D_above, a_num = 0, 0
    # if b_num / D_below >= 0.9:
    #     change_threshold -= del_C
    # # D_below, b_num = 0, 0
    # print(change_threshold)

    response = make_response(str(change_threshold))
    return response

# /high 接口：接收高置信度推理结果; 接收客户端通过 JSON 提交的高置信度框信息；将其直接追加写入 high_img.txt 文件中。
# receive high confidence results
@app.route("/high", methods=["POST"])
def perform_high_images():
    result = json.loads(request.data)
    result_file = open("high_img.txt", "a")
    result_file.write(f"{result['name']}, {result['shape'][0]}, {result['shape'][1]}, {result['shape'][2]}, {result['shape'][3]}, {result['conf']}, {result['label']}\n")
    result_file.close()
    return "save success"


