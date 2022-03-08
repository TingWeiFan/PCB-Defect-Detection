from data_aug import *
import xml.dom.minidom
import os
import numpy as np
import cv2


def save_data(img_, bboxes_, img_name, xml_name, path, aug_name):
    cv2.imwrite('./yolov5-master/data/images/{}_{}'.format(aug_name, img_name), img_)

    p = path[0]
    p.firstChild.data = '/home/n200/datadisk/workspace/elvis/PCB-Defect-Detection/yolov5-master/data/images/{}_{}'.format(aug_name, img_name)

    for j in range(len(bboxes_)):
        x_min = bbox_info[j].getElementsByTagName('xmin')[0]
        y_min = bbox_info[j].getElementsByTagName('ymin')[0]
        x_max = bbox_info[j].getElementsByTagName('xmax')[0]
        y_max = bbox_info[j].getElementsByTagName('ymax')[0]

        x_min.firstChild.data = bboxes_[j][0]
        y_min.firstChild.data = bboxes_[j][1]
        x_max.firstChild.data = bboxes_[j][2]
        y_max.firstChild.data = bboxes_[j][3]

    with open('./yolov5-master/data/Annotations/{}_{}'.format(aug_name, xml_name), 'w') as fh:
        dom.writexml(fh)


img_path = '/home/n200/datadisk/workspace/elvis/PCB-Defect-Detection/yolov5-master/data/images/'
xml_path = '/home/n200/datadisk/workspace/elvis/PCB-Defect-Detection/yolov5-master/data/Annotations/'

imgs = sorted(os.listdir(img_path))
xmls = sorted(os.listdir(xml_path))

for i in range(len(imgs)):
    image = os.path.join(img_path, imgs[i])
    info = os.path.join(xml_path, xmls[i])
    dom = xml.dom.minidom.parse(info)
    root = dom.documentElement

    path = root.getElementsByTagName('path')
    bbox_info = root.getElementsByTagName('object')
    bboxes = []
    for j in range(len(bbox_info)):
        x_min = bbox_info[j].getElementsByTagName('xmin')[0].firstChild.nodeValue
        y_min = bbox_info[j].getElementsByTagName('ymin')[0].firstChild.nodeValue
        x_max = bbox_info[j].getElementsByTagName('xmax')[0].firstChild.nodeValue
        y_max = bbox_info[j].getElementsByTagName('ymax')[0].firstChild.nodeValue
        c = bbox_info[j].getElementsByTagName('name')[0].firstChild.nodeValue
        
        info = np.asarray([float(x_min), float(y_min), float(x_max), float(y_max)])
        bboxes.append(info)

    bboxes = np.asarray(bboxes)
    img = cv2.imread(image)

    # flip
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    save_data(img_, bboxes_, imgs[i], xmls[i], path, 'flip')

    # scale
    img_, bboxes_ = RandomScale(0.3, diff=True)(img.copy(), bboxes.copy())
    save_data(img_, bboxes_, imgs[i], xmls[i], path, 'scale')

    # translation
    img_, bboxes_ = RandomTranslate(0.3, diff=True)(img.copy(), bboxes.copy())
    save_data(img_, bboxes_, imgs[i], xmls[i], path, 'trans')

    # rotation
    img_, bboxes_ = RandomRotate(40)(img.copy(), bboxes.copy())
    save_data(img_, bboxes_, imgs[i], xmls[i], path, 'rotate')

    # shearing
    img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
    save_data(img_, bboxes_, imgs[i], xmls[i], path, 'shear')