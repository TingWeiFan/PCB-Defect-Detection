import os
import cv2
import imutils
from detect import PCB_Net


def iou(bboxs, classes, y, label):
    scores = []
    label_is_true = None
    for pred in bboxs:
        xx1 = max(pred[0], y[0])
        yy1 = max(pred[1], y[1])
        xx2 = min(pred[2], y[2])
        yy2 = min(pred[3], y[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        area = w * h
        score = area / ((pred[2] - pred[0]) * (pred[3] - pred[1]) + (y[2] - y[0]) * (y[3] - y[1]) - area)
        scores.append(score)

    score = max(scores)
    idx = scores.index(score)
    bbox_classes = classes[idx]

    if bbox_classes == label and score >= 0.5:
        score = 1.0
        label_is_true = True
    elif bbox_classes == label and score < 0.5:
        score = score
        label_is_true = True
    elif bbox_classes != label:
        score = 0.0
        label_is_true = False
    else:
        score = 0.0
    return score, label_is_true

net = PCB_Net()
if __name__ == "__main__":
    image_paths = sorted(os.listdir("./yolov5-master/testImages/"))
    #image_paths = ["01_open_circuit_02.jpg"]
    acc = []
    num = 0
    for im in image_paths:
        image_path = os.path.join("./yolov5-master/testImages/", im)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        defect_bbox, defect_classes = net.pcb_defect_detect(image_path)

        image_name = im.split(".")[0]
        f = open("./yolov5-master/data/labels/{}.txt".format(image_name))
        ground_truth = []
        classes = []
        for line in f.read().splitlines(): # 忽略'\n'
            cx, cy = float(line.split(" ")[1]), float(line.split(" ")[2])
            wh, hh = float(line.split(" ")[3])/2, float(line.split(" ")[4])/2
            x0 = int((cx - wh) * w)
            y0 = int((cy - hh) * h)
            x1 = int((cx + wh) * w)
            y1 = int((cy + hh) * h)

            ground_truth.append([x0, y0, x1, y1])
            classes.append(int(line.split(" ")[0]))

        for i in range(len(ground_truth)):
            score, label_is_true = iou(defect_bbox, defect_classes, ground_truth[i], classes[i])
            if label_is_true is not None:
                acc.append(score)
                num += 1
                cv2.rectangle(image, (ground_truth[i][0], ground_truth[i][1]), (ground_truth[i][2], ground_truth[i][3]), (0, 0, 255), 2)
                print("score:{}".format(score))

        for i in range(len(defect_bbox)):
            cv2.rectangle(image, (defect_bbox[i][0], defect_bbox[i][1]), (defect_bbox[i][2], defect_bbox[i][3]), (0, 255, 0), 2)

        cv2.imwrite("./yolov5-master/runs/detect/output/{}".format(im), image)

    accuracy = sum(acc) / num
    print("ACC:{}".format(accuracy))