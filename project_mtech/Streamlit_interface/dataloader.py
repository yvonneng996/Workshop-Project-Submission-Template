# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 18:21:35 2026

@author: Bei Zhen
"""
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

class ForunlabelDataset(Dataset):
    def __init__(self, unlabimgpath_list, batch_size=2, img_sizer=256, img_sizec=256):
        self.unlabimgpath_list=unlabimgpath_list
        self.img_sizer=img_sizer
        self.img_sizec=img_sizec
        self.batch_size=batch_size
    def __len__(self):
        return int(np.ceil(len(self.unlabimgpath_list)/float(self.batch_size)))  
    def __getitem__(self, index):
        
        start = index*self.batch_size
        ## Make temporary variable for batch size
        batch_size = self.batch_size
        ## Calculate the batch size for the LAST data batch
        ## This is because the modulus of the entire data and batch size is not always zero
        if start + self.batch_size > len(self.unlabimgpath_list):
            batch_size = len(self.unlabimgpath_list) - start
        ## Initialize empty image numpy array for one batch
        unlab_img = []
        
        unlabpath_batch= self.unlabimgpath_list[start : start + batch_size]
        #print("a",unlabpath_batch)
        for unlabimgpath in unlabpath_batch:
            unlab = cv2.imread(unlabimgpath)
            unlab=cv2.resize(unlab,(self.img_sizer,self.img_sizec))
            unlab=unlab/255.0
            unlab_img.append(unlab)
            
        unlab_img=np.array(unlab_img)
        unlab_img=np.transpose(unlab_img,(0,3,1,2))
        unlab_img=unlab_img.astype(np.float32)
        unlab_img= torch.from_numpy(unlab_img)
        #print('a',unlab_img.shape)
        return unlab_img