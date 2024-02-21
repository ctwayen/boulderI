import numpy as np
import cv2
import copy

class Process(object):
    suffix = '.jpg'
    def __init__(self):
        pass
    
    def process(self, image_file, shapes):
        start = shapes['起步点']
        end = shapes['结束点']
        crop = shapes['裁剪边框']
        
        src_img = cv2.imread(image_file)
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        print(src_img.shape)
        
        if crop:
            points = crop['points']
            xmin, ymin, xmax, ymax = Process.convertPoints2BndBox(points)
            src_img = src_img[xmin:xmax, ymin:ymax]
        
        start_img = copy.deepcopy(src_img)
        for start_shapes in start:
            points = start_shapes['points']
            xmin, ymin, xmax, ymax = Process.convertPoints2BndBox(points)
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, top_left, bottom_right, color, thickness)
        
    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    @staticmethod
    def resultSave(save_path, image_np):
        cv2.imwrite(save_path, image_np)
        