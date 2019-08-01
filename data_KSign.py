import numpy as np
import json
import os
import scipy.io as sio

from  jsonmerge import merge
from PIL import Image

from ssd_data import BaseGTUtility

class GTUtility(BaseGTUtility):
    """
    Utility for Korean-text dataset.

    """

    def __init__(self, data_path, quality = 'high', polygon=True, only_with_label=True, fi = False):
        self.data_path = data_path
        gt_path = data_path
        image_path = data_path
        #image_path = os.path.join(data_path, 'train')
        self.gt_path = gt_path
        self.image_path = image_path
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        
        if quality == 'high':
            with open(os.path.join(gt_path, 'HighQuality.json'),encoding='UTF8') as f:
                gt_data = json.load(f)
       
        elif quality == 'cr':
            with open(os.path.join(gt_path, 'Annotation.json'),encoding='UTF8') as f:
                gt_data = json.load(f)
                
        for img in gt_data.keys():
            image_name = gt_data[img]['filename']
            #print(image_name)
            
            boxes = []
            text = []
            for ann in range(len(gt_data[img]['regions'])):
                gt_type = gt_data[img]['regions'][ann]['shape_attributes']['name']

                if gt_type == "rect":
                    x = gt_data[img]['regions'][ann]['shape_attributes']['x']
                    y = gt_data[img]['regions'][ann]['shape_attributes']['y']
                    w = gt_data[img]['regions'][ann]['shape_attributes']['width']
                    h = gt_data[img]['regions'][ann]['shape_attributes']['height']
                    
                    #box = np.array([x,y, x+w, y+h], dtype=np.float32)
                    #box = np.array([x,y,x,y+h,x+w,y+h,x+w,y], dtype=np.float32)
                    box = np.array([x,y,x+w,y,x+w,y+h,x,y+h], dtype=np.float32)
                    
                elif gt_type == "polygon": # for 4points
                    xlist = gt_data[img]['regions'][ann]['shape_attributes']['all_points_x']
                    ylist = gt_data[img]['regions'][ann]['shape_attributes']['all_points_y']
                    
                    if len(xlist)!= 4:
                        continue
                    
                    xy= []
                    xy = [xlist[0]]+ [ylist[0]]+ \
                         [xlist[3]]+ [ylist[3]]+ \
                         [xlist[2]]+ [ylist[2]]+ \
                         [xlist[1]]+ [ylist[1]]
                    #여기서 박스의 모양:
                    #가장 왼쪽 상단 > 오른쪽 상단 > 오른쪽 하단 > 왼쪽 하단 순서로 이동.
                    
                    '''for x, y in zip(xlist, ylist):
                        xy = xy + [x] + [y]'''
                    
                    box = np.array(xy, dtype=np.float32)
                else:
                    continue
                    
                if 'annotation_kr' in gt_data[img]['regions'][ann]['region_attributes'].keys():
                    txt = gt_data[img]['regions'][ann]['region_attributes']['annotation_kr']
                else:
                    if only_with_label:
                        continue
                    else:
                        txt = ''
                        
                boxes.append(box)
                text.append(txt)

            if len(boxes) == 0:
                continue
            
            boxes = np.asarray(boxes)
           
            
            with Image.open(os.path.join(data_path,image_name)) as img:
                width, height = img.size
            
            boxes[:,0::2] /= width
            boxes[:,1::2] /= height
            
            if filter:
                eps = 1e-3

                p1, p2, p3, p4 = boxes[:,0:2], boxes[:,2:4], boxes[:,4:6],boxes[:,6:8]
                # fix twisted boxes
                if True:
                    mask = np.linalg.norm(p1 + p2 - p3 - p4, axis=1) < eps
                    boxes[mask] = np.concatenate([p1[mask], p3[mask], p2[mask], p4[mask]], axis=1)
                # filter out bad boxes
                if True:
                    mask = np.ones(len(boxes), dtype=np.bool)
                    # filter boxes with zero width
                    boxes_w = np.linalg.norm(p1-p2, axis=1)
                    boxes_h = np.linalg.norm(p2-p3, axis=1)
                    mask = np.logical_and(mask, boxes_w > eps)
                    mask = np.logical_and(mask, boxes_h > eps)
                    # filter boxes that are too large
                    mask = np.logical_and(mask, np.all(boxes > -1, axis=1))
                    mask = np.logical_and(mask, np.all(boxes < 2, axis=1))
                    # filter boxes with all vertices outside the image
                    boxes_x = boxes[:,0::2]
                    boxes_y = boxes[:,1::2]
                    mask = np.logical_and(mask, 
                            np.sum(np.logical_or(np.logical_or(boxes_x < 0, boxes_x > 1), 
                                    np.logical_or(boxes_y < 0, boxes_y > 1)), axis=1) < 4)
                    # filter boxes with center outside the image
                    boxes_x_mean = np.mean(boxes[:,0::2], axis=1)
                    boxes_y_mean = np.mean(boxes[:,1::2], axis=1)
                    mask = np.logical_and(mask, np.logical_and(boxes_x_mean > 0, boxes_x_mean < 1))
                    mask = np.logical_and(mask, np.logical_and(boxes_y_mean > 0, boxes_y_mean < 1))
                    boxes = boxes[mask]
                    text = np.asarray(text)[mask]
                
                if len(boxes) == 0:
                    continue
            
            boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
            
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)
        
        self.init()
        
