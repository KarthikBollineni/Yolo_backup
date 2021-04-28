#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:11:20 2021

@author: vstnh-lap12
"""






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:04:12 2021

@author: vstnh-lap12
"""


import cv2
import imutils
import numpy as np
from PIL import Image
import cv2 as cv
from detecto.core import Model
from detecto import utils, visualize
import matplotlib.pyplot as plt
from detecto.core import Model
from detecto.utils import read_image
import cv2


model = Model.load('/home/vstnh-lap12/Downloads/Damage.pth', ['Dent','scratches','Dent_with_heavy_cracks','Dent_with_cracks','Windshield_heavy_damage','Windshield_crack','heavy_dents','tail_light_left','tail_light_right','Head_light_left','Head_light_left','Windshield_crack_spot','hood_bend'])
#cap = cv2.VideoCapture("/content/4.jpg")
cap = cv2.VideoCapture("/home/vstnh-lap12/Downloads/VIDEO_21.mp4")
print('ok')


while True:
    #get frame from video
    hasFrame, frame = cap.read()
    print('nice')
    if hasFrame==True:
        print('ok')
        frame1=frame
        labels, boxes, scores = model.predict(frame1)
        
        top_preds = model.predict_top(frame)
        print(top_preds)
        if len(top_preds)>0:
            try:
                y,x,h,w=int(top_preds[1][0][0]),int(top_preds[1][0][1]),int(top_preds[1][0][2]),int(top_preds[1][0][3])
        
                crop_image = frame[x:w, y:h]
                plt.imshow(crop_image, cmap="hot")

            except Exception:
                pass
    else:
        print('ok')
        break