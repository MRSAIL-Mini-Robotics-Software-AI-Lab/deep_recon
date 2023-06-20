#!/usr/bin/env python3



import argparse
import cv2
import math
from models import hsm
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform



class HSMInterface():
    def __init__(self,max_disparity,level,clean,testres=1) -> None:
        self.max_disparity = max_disparity
        self.level = level
        self.clean = clean
        self.testres = testres
        self.model = hsm(128,1.0,level=1)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.processed = get_transform()
        tmpdisp = int(self.max_disparity*testres//64*64)
        if (self.max_disparity*testres/64*64) > tmpdisp:
            self.model.module.maxdisp = tmpdisp + 64
        else:
            self.model.module.maxdisp = tmpdisp
        if self.model.module.maxdisp ==64: self.model.module.maxdisp=128
        self.model.module.disp_reg8 =  disparityregression(self.model.module.maxdisp,16)
        self.model.module.disp_reg16 = disparityregression(self.model.module.maxdisp,16)
        self.model.module.disp_reg32 = disparityregression(self.model.module.maxdisp,32)
        self.model.module.disp_reg64 = disparityregression(self.model.module.maxdisp,64)

    def load_model(self,model_path):
        pretrained_dict = torch.load(model_path,map_location=torch.device('cpu'))
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        self.model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        self.model.eval()
        print('Model loaded')
        
    def test(self,leftImage,rightImage):
        imgL_o = cv2.resize(leftImage,None,fx=self.testres,fy=self.testres,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(rightImage,None,fx=self.testres,fy=self.testres,interpolation=cv2.INTER_CUBIC)
        imgL = self.processed(imgL_o).numpy()
        imgR = self.processed(imgR_o).numpy()
        imgsize = imgL_o.shape[:2]
        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        with torch.no_grad():
            start_time = time.time()
            pred_disp,entropy = self.model(imgL,imgR)
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

        pred_disp = cv2.resize(pred_disp/self.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip and replace inf with 0
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = 0

        return pred_disp,entropy
    


if __name__ == "__main__":
    hsm = HSMInterface(256,1.0,1.0)
    hsm.load_model('src/weights/final-768px.tar')
    imgL_o = cv2.imread('src/left0000.jpg')
    imgR_o = cv2.imread('src/right0000.jpg')
    pred_disp,entropy = hsm.test(imgL_o,imgR_o)
    cv2.imwrite('disp-1024.png',pred_disp/pred_disp.max()*255)