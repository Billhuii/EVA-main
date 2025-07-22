import os
import time
import math
import cv2 as cv
import json
import requests
import random

# ---------------------- 全局变量配置 ----------------------

# 服务端地址（默认本地运行 Flask）
HNAME = '127.0.0.1:5001'

# 图像数据路径（原始图像帧）
DATAPATH = './dataset/trafficcam_2/src/'

# 生命周期：目标从首次检测到必须上报的最长容忍时间（单位：毫秒）
T = 10000

# 预估网络延迟（单位：毫秒）
L = 500

# 网络最大传输带宽（单位：bytes/ms）
R = 10000

# 带宽调节因子（带宽上限 = (剩余时间 - 延迟) × fB）
fB = 2

# 最大可延迟上报的时间（越过即需强制上传图像）
AGE = T - L - R / fB

# ---------------------- 高置信度目标上传 ----------------------

def send_to_server(target):
    """
    将高置信度目标的结构信息以 JSON 格式发送至服务器端的 /high 接口。
    不发送图像，只发送目标位置、标签、置信度等数据。
    """
    to_send = {
        'name': target['name'],
        'shape': target['shape'],
        'conf': target['confidence'],
        'label': target['result'],
        'capture_timestamp': target['birth']
    }
    response = requests.Session().post(
        "http://" + HNAME + "/high", data=json.dumps(to_send)
    )
    print('Successfully sent result ', target['name'], ' to server!')

# ---------------------- 低置信度目标图像上传 ----------------------

def send_image_to_server(target, age, bw):
    """
    将低置信度目标的图像压缩后发送至服务器端的 /low 接口。
    根据目标“年龄”限制传输速率，并统计带宽使用量。
    """
    # 动态调节目标允许的最大带宽（越“老”越小）
    r = (T - age - L) * fB
    r = min(r, R)  # 限制最大值

    # 从 cache 中读取缓存图像
    raw_path = './cache/' + target['name']
    raw = cv.imread(raw_path)

    # 将图像压缩后存入临时目录
    temp_path = './temp/' + target['name']
    cv.imwrite(temp_path, raw, [cv.IMWRITE_PNG_COMPRESSION, 9])

    # 累加带宽消耗
    to_send_file = {'image': open(temp_path, 'rb')}
    bw += os.path.getsize(temp_path)

    print("Accumulated Bandwidth: ", bw)

    # 组装发送内容（含结构数据 + 压缩图像）
    to_send_data = {
        "name": target['name'],
        "shape": target['shape'],
        'conf': target['confidence'],
        'label': target['result'],
        'bw': os.path.getsize(temp_path),
        'capture_timestamp': target['birth']
    }

    response = requests.Session().post(
        "http://" + HNAME + "/low", files=to_send_file, data=to_send_data
    )

    # 服务器返回新的置信度阈值（可能为 0.0 表示不变）
    nC = float(response.content)
    print('Successfully sent image ', target, ' to server with r as ' + str(r) + ' (max=10000).')

    return bw, nC

# ---------------------- 缓存图像的生成 ----------------------

def cache_append(img):
    """
    从原始图像中裁剪出目标区域并保存到 ./cache 目录，用于后续上传。
    """
    position = img['shape']
    raw_path = DATAPATH + img['name'][-14:]  # 获取图像原始路径
    raw = cv.imread(raw_path)

    # 裁剪目标区域
    segment = './cache/' + img['name']
    a = raw[position[0]:position[2], position[1]:position[3]]

    # 保存图像（无压缩）供后续使用
    cv.imwrite(segment, a, [cv.IMWRITE_PNG_COMPRESSION, 0])
    print('Picture ', img['name'], ' appended into cache!')

# ---------------------- 移除缓存图像 ----------------------

def cache_pop(img):
    """
    删除 cache 中对应名称的图像。
    """
    img_path = './cache/' + img
    os.remove(img_path)
    print('Picture', img, ' popped from cache!')

# ---------------------- 年龄计算 ----------------------

def check_age(target):
    """
    返回目标“出生”至当前的毫秒差值（考虑周期性 T）。
    """
    age = math.floor(time.time() * 1000 % T) - int(target['birth'])
    return age if age > 0 else age + T  # 防止负数，模 T 后保证为正时间差

# ---------------------- 在目标列表中查找并返回目标 ----------------------

def find_target(targets, name):
    """
    在目标列表中按名称查找目标，并返回找到的目标（同时从列表中移除）。
    """
    for index, target in enumerate(targets):
        if target['name'] == name:
            print("Found target: ", target)
            return targets.pop(index)
    return -1

# ---------------------- 图像帧采样函数（可选） ----------------------

def generate_data():
    """
    从图像序列中随机抽取一个连续段，随机丢弃一半帧，模拟缺帧情况。
    用于测试或模拟稀疏检测。
    """
    raw = os.listdir("./dataset/trafficcam_2/src")
    raw.sort()

    start = random.randint(0, 30)  # 随机起点
    raw = raw[start:start + 30]

    # 去除偶数下标帧（留下奇数帧）
    rm = [i * 2 for i in range(15)]
    gen = [i for num, i in enumerate(raw) if num not in rm]

    return gen

