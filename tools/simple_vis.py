import cv2
import os
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def open_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        return lines

def get_obj(txt_list):
    res = []
    for line in txt_list:
        temp = line.strip("\n").split(" ")
        if temp[0] in labels:
            res.append([temp[0], temp[4], temp[5], temp[6], temp[7]])
    return res

def draw_obj(img, obj_list, col):
    for obj in obj_list:
        cls = obj.pop(0)
        obj = list(map(int, map(eval, obj)))
        cv2.rectangle(img, 
                      (int(obj[0]),int(obj[1])), 
                      (int(obj[2]),int(obj[3])), 
                      col,
                      1)
        cv2.putText(img, cls, (int(obj[0]),int(obj[1])), font, .5, col)

gt_txt_dir = '/home/ubuntu/Data/kitti/training/label_2'
pre_txt_dir = '/home/ubuntu/Data/kitti/training/label_2'
img_dir = '/home/ubuntu/Data/kitti/training/image_2'
labels = ["Car", "Cyclist", "Pedestrian"]
has_gt = True
for pre_txt_name in os.listdir(pre_txt_dir):
    if os.path.splitext(pre_txt_name)[1] == '.txt':
        pre_txt_path = os.path.join(pre_txt_dir, pre_txt_name)
        pre_txt = open_txt(pre_txt_path)
        objs = get_obj(pre_txt)
        img = cv2.imread(os.path.join(img_dir, pre_txt_name.replace('txt', 'png')))
        draw_obj(img, objs, (0,255,0))
        cv2.waitKey(10)
        if has_gt:
            gt_txt_path = os.path.join(gt_txt_dir, pre_txt_name)
            gt_txt = open_txt(gt_txt_path)
            objs = get_obj(gt_txt)
            draw_obj(img, objs, (0,0,255))
        cv2.imshow('', img)
        cv2.waitKey(10)
