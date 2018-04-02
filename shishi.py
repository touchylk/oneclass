# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import cPickle
import datetime
import os
import yolo.config as cfg
import matplotlib.pyplot as plt
from yolo.yolo_net import YOLONet
from utils.timer import Timer
import utils.ap_voc as voc

data_dir = '/media/e813/E/CarPic/'

"""for d_1 in os.listdir(data_dir):
    abd_1 = os.path.join(data_dir,d_1)
    if os.path.isdir(abd_1):
        for d_2 in os.listdir(abd_1):

            abd_2 = os.path.join(abd_1,d_2)
            if os.path.isdir(abd_2):
                for d_3 in os.listdir(abd_2):
                    abd_3 = os.path.join(abd_2,d_3)
                    #print(abd_3)
                    if os.path.isdir(abd_3):
                        for d_4 in os.listdir(abd_3):
                            abd_4 = os.path.join(abd_3,d_4)
                            #print(abd_4)
                            if abd_4[-3:] == 'xml':
                                print(abd_4)
                    elif abd_3[-3:]=='xml':
                        print(abd_3)
            elif abd_2[-3:]=='xml':
                print(abd_2)
    elif abd_1[-3:]=='xml':
        print(abd_1)"""
a = {'aa':1,'bb':2}
print(a['aa'])
if 'cc' not in a:
    print(a['bb'])
else:
    print(a['aa'])




