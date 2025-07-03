import shutil
import time
import math
import yolo
import trace
import utils
import os
import csv

# 目标的生命周期周期（以毫秒为单位）
T = 10000

# 初始置信度阈值
C = 0.8

# 原始数据集路径
img_path = os.path.join("dataset/trafficcam_2/src")

def main():
    global C  # 置信度阈值可能会被动态调整

    # 初始化目录与文件：删除旧缓存，创建新缓存目录
    for direc in ['./cache', './temp']:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.mkdir(direc)

    # 清空历史带宽记录文件
    for file in ['./result_bw.csv']:
        if os.path.isfile(file):
            os.remove(file)
        f = open(file, 'w+')
        f.close()

    last_frame = []  # 上一帧的检测结果，用于目标追踪

    # 获取数据集中所有帧图像
    raw_set = os.listdir('./dataset/trafficcam_2/src')
    raw_set.sort()

    for raw_img in raw_set:
        bw = 0  # 每一帧图像传输带宽计数器
        raw_path = os.path.join(img_path, raw_img)  # 当前帧的完整路径

        # YOLO 推理获得当前帧的检测目标
        # 如raw_current = [
        #     [0.32, 0.4, 0.1, 0.15, 0.82, 'car'],
        #     [0.12, 0.18, 0.05, 0.07, 0.76, 'bus'],
        #     [0.6, 0.5, 0.2, 0.25, 0.44, 'person']
        # ]
        raw_current = yolo.detect(raw_path)
        current_frame = []

        # 打印上一次帧的检测信息（如果存在）
        if len(last_frame) > 0:
            print('\n' + '-' * 90 + '\n')
            for lt in last_frame:
                print(lt)

        print('\nCURRENT TARGETS：')
        # 构建当前帧目标信息字典
        # 如[
        #     {'name': '0_frame1.jpg', 'shape': (100, 50, 40, 40), 'confidence': 0.75, 'result': 'car', 'detected': False, 'birth': 120},
        #     {'name': '1_frame1.jpg', 'shape': (200, 100, 30, 30), 'confidence': 0.60, 'result': 'person', 'detected': False, 'birth': 121},
        # ]
        for index, target in enumerate(raw_current):    # 每一个target为一个列表，如[0.32, 0.4, 0.1, 0.15, 0.82, 'car']
            t = {
                'name': f"{index}_{raw_img}",         # 唯一标识名
                'shape': target[:4],                  # 位置信息 (x, y, w, h)
                'confidence': target[4],              # 置信度
                'result': target[5],                  # 类别标签
                'detected': False,                    # 是否已发送至服务端
                'birth': math.floor(time.time() * 1000 % T)  # 生成时间戳（取模生命周期）
            }
            current_frame.append(t)
            print('target', index, ':', t)

        # 使用 trace 模块进行 IoU 匹配以追踪上一帧中的目标
        # tracked: 得到匹配对 [(i,j), ...] 表示last和current的index对应
        last_frame, current_frame, tracked = trace.preprocess_data(last_frame, current_frame)
        print('\nTracked.\n')

        put_back = []

        # 遍历匹配到的目标对（跨帧）
        if len(tracked) > 0:
            print('Found same targets:', '\n', tracked)
            for pair in tracked:
                last_target = utils.find_target(last_frame, pair[0])
                temp_target = utils.find_target(current_frame, pair[1])

                # last_target和temp_target是同一个目标
                # 如果上一个目标还没发送
                if not last_target['detected']:
                    if float(temp_target['confidence']) >= C:
                        # 满足置信度直接发送
                        print('Because of enough CONFIDENCE,')
                        utils.cache_append(temp_target)
                        utils.cache_pop(last_target['name'])

                        # 接近阈值的目标进一步提交服务端判断是否调整阈值
                        if float(temp_target['confidence']) - C < 0.05:
                            bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
                            if nC != 0.0:
                                C = nC
                        else:
                            utils.send_to_server(temp_target)

                        temp_target['detected'] = True  # 表示temp_target已发送
                        put_back.append(temp_target)
                    else:
                        # 本帧置信度更高则替换旧目标
                        if float(temp_target['confidence']) > float(last_target['confidence']):
                            print('Because of higher confidence,')
                            # 目标真正首次出现的时间不能变，不如会超过AGE还不发送
                            temp_target['birth'] = last_target['birth']
                            utils.cache_append(temp_target)
                            utils.cache_pop(last_target['name'])
                            # 系统中设置了一个最大寿命（如 AGE = 500 毫秒），过了这个寿命仍未发送，就必须强制发送
                            if utils.check_age(last_target) > utils.AGE:
                                print('Because of time limit,')
                                bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
                                if nC != 0.0:
                                    C = nC
                                temp_target['detected'] = True
                            put_back.append(temp_target)
                        else:
                            # 本帧置信度低于上一帧：发送旧目标
                            print('Because of lower confidence,')
                            bw, nC = utils.send_image_to_server(last_target, utils.check_age(last_target), bw)
                            if nC != 0.0:
                                C = nC
                            last_target['detected'] = True
                            put_back.append(last_target)
                else:
                    # 已发送的目标保持跟踪状态
                    print('Save detected target ', temp_target['name'], ' .')
                    temp_target['detected'] = True
                    utils.cache_append(temp_target)
                    utils.cache_pop(last_target['name'])
                    put_back.append(temp_target)

        # 当前帧中丢失的目标：生命周期满则强制发送
        for lost_target in last_frame:
            print('Because of target lost,')
            if not lost_target['detected']:
                bw, nC = utils.send_image_to_server(lost_target, utils.check_age(lost_target), bw)
                if nC != 0.0:
                    C = nC
            utils.cache_pop(lost_target['name'])

        # 当前帧中新发现的目标（未被跟踪）
        for index, new_found in enumerate(current_frame):
            print('New target detected:\n', new_found)
            utils.cache_append(new_found)
            if float(new_found['confidence']) >= C:
                print('Because of enough CONFIDENCE,')
                utils.send_to_server(new_found)
                new_found['detected'] = True

        # 当前帧的总目标状态：历史+新发现
        current_frame.extend(put_back)
        last_frame = current_frame

        # 写入每帧的带宽使用信息
        results_files = open("result_bw.csv", "a")
        csv_writer = csv.writer(results_files)
        csv_writer.writerow([bw])
        results_files.close()

        print(bw / 1024)  # 带宽使用 KB 输出

if __name__ == '__main__':
    main()



# import shutil
# import time
# import math
# import yolo
# import trace
# import utils
# import os
# import csv
#
# # Lifetime for a target, detected targets will be sent to server within its lifetime
# T = 10000
# # Confidence threshold
# C = 0.8
# # Dataset path
# img_path = os.path.join("dataset/trafficcam_2/src")
#
#
# def main():
#     global C
#     for direc in ['./cache', './temp']:
#         if os.path.exists(direc):
#             shutil.rmtree(direc)
#         os.mkdir(direc)
#     for file in ['./result_bw.csv']:
#         if os.path.isfile(file):
#             os.remove(file)
#         f = open(file, 'w+')
#         f.close()
#     last_frame = []
#     # Origin Dateset
#     raw_set = os.listdir('./dataset/trafficcam_2/src')
#     raw_set.sort()
#     # Sampling Function
#     # raw_set = utils.generate_data()
#     for raw_img in raw_set:
#         bw = 0
#         raw_path = os.path.join(img_path, raw_img)
#         raw_current = yolo.detect(raw_path)
#         current_frame = []
#         if len(last_frame) > 0:
#             print(
#                 '\n-----------------------------------------------------------------------------------------\n')
#             for lt in last_frame:
#                 print(lt)
#         print('\nCURRENT TARGETS：')
#         for index, target in enumerate(raw_current):
#             t = {'name': str(index) + '_' + raw_img, 'shape': target[:4], 'confidence': target[4],
#                  'result': target[5],
#                  'detected': False,
#                  'birth': math.floor(time.time() * 1000 % T)}
#             current_frame.append(t)
#             print('target', index, ':', t)
#         last_frame, current_frame, tracked = trace.preprocess_data(last_frame, current_frame)
#         print('\nTracked.\n')
#         put_back = []
#         if len(tracked) > 0:
#             print('Found same targets:', '\n', tracked)
#             for pair in tracked:
#                 last_target = utils.find_target(last_frame, pair[0])
#                 temp_target = utils.find_target(current_frame, pair[1])
#                 if not last_target['detected']:
#                     if float(temp_target['confidence']) >= C:
#                         print('Because of enough CONFIDENCE,')
#                         utils.cache_append(temp_target)
#                         utils.cache_pop(last_target['name'])
#                         # Send targets which are slightly upon C to server entirely to adjust confidence threshold
#                         if float(temp_target['confidence']) - C < 0.05:
#                             bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
#                             if not nC == 0.0:
#                                 C = nC
#                         else:
#                             utils.send_to_server(temp_target)
#                         temp_target['detected'] = True
#                         put_back.append(temp_target)
#                     else:
#                         if float(temp_target['confidence']) > float(last_target['confidence']):
#                             print('Because of higher confidence,')
#                             temp_target['birth'] = last_target['birth']
#                             utils.cache_append(temp_target)
#                             utils.cache_pop(last_target['name'])
#                             if utils.check_age(last_target) > utils.AGE:
#                                 print('Because of time limit,')
#                                 bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
#                                 if not nC == 0.0:
#                                     C = nC
#                                 temp_target['detected'] = True
#                             put_back.append(temp_target)
#                         else:
#                             print('Because of lower confidence,')
#                             bw, nC = utils.send_image_to_server(last_target, utils.check_age(last_target), bw)
#                             if not nC == 0.0:
#                                 C = nC
#                             last_target['detected'] = True
#                             put_back.append(last_target)
#                 else:
#                     print('Save detected target ', temp_target['name'], ' .')
#                     temp_target['detected'] = True
#                     utils.cache_append(temp_target)
#                     utils.cache_pop(last_target['name'])
#                     put_back.append(temp_target)
#         for lost_target in last_frame:
#             print('Because of target lost,')
#             if not lost_target['detected']:
#                 bw, nC = utils.send_image_to_server(lost_target, utils.check_age(lost_target), bw)
#                 if not nC == 0.0:
#                     C = nC
#             utils.cache_pop(lost_target['name'])
#         for index, new_found in enumerate(current_frame):
#             print('New target detected:\n', new_found)
#             utils.cache_append(new_found)
#             if float(new_found['confidence']) >= C:
#                 print('Because of enough CONFIDENCE,')
#                 utils.send_to_server(new_found)
#                 new_found['detected'] = True
#         current_frame.extend(put_back)
#         # Send all cached targets before ending the process
#         # if raw_img == raw_set[-1]:
#         #     for target in current_frame:
#         #         if not target['detected']:
#         #             print("Because it's the last frame,")
#         #             bw, nC = utils.send_image_to_server(target, utils.check_age(last_target), bw)
#         # else:
#         last_frame = current_frame
#         # write bw
#         results_files = open("result_bw.csv", "a")
#         csv_writer = csv.writer(results_files)
#         csv_writer.writerow([bw])
#         results_files.close()
#         print(bw / 1024)
#
#
# if __name__ == '__main__':
#     main()
