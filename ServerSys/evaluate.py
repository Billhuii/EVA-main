from backend.server import Server
from utils import merge_boxes_in_results
import os
import random
server = Server()
THRESHOLD = 0.3


def calc_intersection_area(a, b):
    to = max(a[1], b[1])
    le = max(a[0], b[0])
    bo = min(a[3], b[3])
    ri = min(a[2], b[2])

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a[2] - a[0])
    h = max(0, a[3] - a[1])

    return w * h

# 返回两个框的交并比（IoU），用于后续匹配
def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


# generate ground truth 生成 Ground Truth 标注文件
def gt():
    bandwidth = 0
    # --- MODIFIED: Corrected the dataset path to be inside CameraSys ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_direc = os.path.join(project_root, "CameraSys/dataset/trafficcam_2/src/")
    results_path = os.path.join(project_root, "trafficcam_2_gt")
    # --- End of Modification ---
    results_file = open(results_path, "w")
    for i in range(len(os.listdir(image_direc))):
        fname = f"{str(i).zfill(10)}.png"
        image_path = os.path.join(image_direc, fname) # Now uses absolute path
        bandwidth = bandwidth + os.path.getsize(image_path)
        results, rpn_results = server.perform_detection(image_direc, 1.0, fname)
        results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)
        for region in results.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf}\n")
            results_file.write(str_to_write)

    results_file.close()


# evaluate eva system 读取 GT 与检测结果，计算准确率（precision）
def eva():
    # --- MODIFIED: Use absolute paths for robustness ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_path = os.path.join(project_root, "trafficcam_2_gt")
    cova_path = os.path.join(project_root, "ServerSys/backend/high_img.txt")
    # --- End of Modification ---
    
    gt_box = {}
    cova_box = {}
    # get gt box
    gt_file = open(gt_path, "r")
    for line in gt_file:
        frame = line.split(",")[0]
        x = line.split(",")[1]
        y = line.split(",")[2]
        w = line.split(",")[3]
        h = line.split(",")[4]
        y1 = int(720 * float(y))
        x1 = int(1280 * float(x))
        y2 = int(720 * (float(y) + float(h)))
        x2 = int(1280 * (float(x) + float(w)))
        box = [y1, x1, y2, x2]
        if gt_box.get(int(frame)):
            gt_box[int(frame)].append(box)
        else:
            gt_box[int(frame)] = [box]
    gt_file.close()
    # get cova box
    cova_file = open(cova_path, "r")
    for line in cova_file:
        line = line.strip()
        frame = int(line.split(",")[0].split(".")[0].split("_")[1])
        y1 = int(line.split(",")[1])
        x1 = int(line.split(",")[2])
        y2 = int(line.split(",")[3])
        x2 = int(line.split(",")[4])
        conf = float(line.split(",")[5])
        label = line.split(",")[6].strip()
        box = [y1, x1, y2, x2, line.split(",")[0]]
        if label in ["car",  "motorbike", "vehicle", "bicycle"]:
            if cova_box.get(int(frame)):
                cova_box[int(frame)].append(box)
            else:
                cova_box[int(frame)] = [box]
    cova_file.close()
    # calculate acc
    tp = 0
    fp = 0
    count = []
    for key, value in cova_box.items():
        for box in cova_box[key]:
            found = False
            # ADDED: Check if the frame key exists in ground truth to avoid errors
            if key not in gt_box:
                fp += 1
                count.append(0)
                continue
            for gbox in gt_box[key]:
                if calc_iou(box[:4], gbox[:4]) > THRESHOLD:
                    found = True
                    break
            if found:
                tp += 1
                count.append(1)
            else:
                print(box[4])
                fp += 1
                count.append(0)

    # ADDED: Logic to calculate False Negatives (FN)
    fn = 0
    # Create a copy of cova_box to mark matched boxes, preventing double counting
    cova_box_copy = {k: [item[:] for item in v] for k, v in cova_box.items()}

    for key, value in gt_box.items():
        for gbox in gt_box[key]:
            found = False
            if key in cova_box_copy:
                for cbox in cova_box_copy[key]:
                    if calc_iou(gbox[:4], cbox[:4]) > THRESHOLD:
                        found = True
                        # Remove the matched box so it cannot be matched again
                        cova_box_copy[key].remove(cbox)
                        break
            if not found:
                fn += 1 # This ground truth box was not detected

    # ADDED: Calculate Precision, Recall, and F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # MODIFIED: Print all evaluation metrics
    print("-" * 30)
    print("Evaluation Results:")
    print(f"True Positives (TP):    {tp}")
    print(f"False Positives (FP):   {fp}")
    print(f"False Negatives (FN):   {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("-" * 30)


# gt()
eva()
