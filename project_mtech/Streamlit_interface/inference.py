# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 18:33:29 2026

@author: Bei Zhen
"""

from natsort import natsorted
import os
from model_unet import UNet
import torch
import cv2
import numpy as np
import joblib
from pathlib import Path

def run_unet_inference(img,
    #model,
    img_size=224,
    device="cpu"):
    
    BASE_DIR = Path(__file__).parent
    model_path = BASE_DIR / "Best_Model" / "BestModel_iter136080_iou0.4913.pth"
    model1 = UNet(in_chns=3, class_num=1).cuda()
    model1.load_state_dict(torch.load(model_path)) 
    model1 = model1.to('cpu')
    model1.eval()
#img=natsorted(os.listdir(val_imgpath_list))
#mask=natsorted(os.listdir(val_labpath_list))
    #for a in img:    
    val_imgsrgb = img.copy()
    print("v1",val_imgsrgb.shape)
        
        #cv2.imread(a)
    #val_imgsrgb = cv2.cvtColor(val_imgsrgb, cv2.COLOR_BGR2RGB)  # ensure
    #splits=os.path.split(c)
    #val_ID=splits[1]
    #print(splits)
    val_imgs = cv2.resize(val_imgsrgb,(224,224))
    val_imgs = val_imgs.astype(np.float32) /255.0#np.array(val_imgs)
    val_imgs = np.expand_dims(val_imgs,axis=0)
    val_imgs = np.transpose(val_imgs,(0,3,1,2))
    val_imgs = torch.tensor(val_imgs, dtype=torch.float32)
    print("vi",val_imgs.shape)
        
    
    val_result=model1(val_imgs)
    val_result = torch.sigmoid(val_result)
    val_result = val_result > 0.5
    val_result = val_result.squeeze().cpu().numpy()

    #ret, val_lab_thresh = cv2.threshold(val_labsgray, 127, 255, 0)
    #val_lab_contours, h = cv2.findContours(val_lab_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    val_pred = np.reshape(val_result.astype('uint8')*255, (img_size, img_size))
    val_pred = cv2.resize(val_pred,(val_imgsrgb.shape[1],val_imgsrgb.shape[0]))
    
    ret, val_pred_thresh = cv2.threshold(val_pred, 127, 255, 0)
    val_pred_contours, h = cv2.findContours(val_pred_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    val_imgtemp = 1*val_imgsrgb
    #cv2.drawContours(val_imgtemp, val_lab_contours, -1, (0,255,0), 2, cv2.LINE_8)
    b=cv2.drawContours(val_imgtemp, val_pred_contours, -1, (0,0,255), 2, cv2.LINE_8)
    #cv2.imwrite("D:\\newly vessel train\\test newly preprocess\\prediction train with only mina\\try\\", val_pred)
    #cv2.imwrite(save_valimage + val_ID, val_imgtemp)
    print("val_pred shape:", val_pred.shape)
    return val_pred