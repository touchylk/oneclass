# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import datetime
#
# path and dataset parameter
#
LAST_STEP = 0
AX_LOW = 1.0
AX_HIGHT = 40

GPU = '1,0'

PASCAL_PATH = '/home/e813/dataset/VOCdevkit_2012_trainval/'
PASCAL_dir_test = '/home/e813/dataset/VOCdevkit_2012_trainval/VOC2012/'
#PASCAL_dir_test = '/home/e813/dataset/VOCdevkit_2007_test/VOC2007/'


OUTPUT_dir = 'data/output'

LAYER_to_restore = 46
WEIGHTS_init_dir_file = '/home/e813/yolo_weights/YOLO_small.ckpt'
WEIGHTS_output_dir = '/media/e813/D/weight_output1/oneclass/BN_noinit'
TRAIN_process_save_txt_dir = WEIGHTS_output_dir
WEIGHTS_file_name = 'save.ckpt-{}'.format(LAST_STEP)
#WEIGHTS_to_restore = WEIGHTS_init_dir_file
WEIGHTS_to_restore = WEIGHTS_init_dir_file #os.path.join(WEIGHTS_output_dir,WEIGHTS_file_name) #
WEIGHTS_to_save_dir = WEIGHTS_output_dir



CACHE_PATH = 'data/label_cache'
WERIGHTS_READ = os.path.join(WEIGHTS_output_dir,WEIGHTS_file_name)
#WERIGHTS_READ = '/home/e813/yolo_weights/output/yolo_verclass/save.ckpt-6000'

IMAGE_dir_file = '/home/e813/dataset/VOCdevkit_2007_test/VOC2007/JPEGImages/000979.jpg'
#IMAGE_dir_file = '/home/e813/dataset/VOCdevkit _2007_trainval/VOC2007/JPEGImages/000021.jpg'

if 0:
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
elif 0:
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
elif 0:
    CLASSES = ['person', 'car', 'dog', 'chair']
else:
    CLASSES = ['person']

CLASS_NUM = len(CLASSES)
if CLASS_NUM == 1:
    ONE_CLASS = True
else:
    ONE_CLASS = False

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.1
CLASS_SCALE = 1.0
COORD_SCALE = 5.0


#
# solver parameter
#



LEARNING_RATE = 0.002

DECAY_STEPS = 500

DECAY_RATE = 0.316

STAIRCASE = True

BATCH_SIZE = 20

MAX_ITER = 1000

SUMMARY_ITER = 2

SAVE_ITER = 200


#
# test parameter
#


TEST_batch_size = 40

#THRESHOLD = 0.3

C_THRESHOLD = 0.5
TEST_IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5

"""
1,文件记录每一次训练用到了第几张图片,每一个epoch,将shuffle后的标签储存,使得每一次重新训练都能像继续训练一样
2,学习结构化保存对象的包,储存训练的每一个过程.
3,双GPU分配batch
"""