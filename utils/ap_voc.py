# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy
import yolo.config as cfg


class label(object):
    def __init__(self, image_path, label_path):
        self.image_dir_file = image_path
        self.label_dir_file = label_path
        self.label_content = []
        self.box_num = 0
        self.obj_num = 0
        self.is_person = False
        self.init_label()

    def init_label(self):
        """
        box,归一化后的标签值.
        :return:
        """
        im = cv2.imread(self.image_dir_file)
        h = im.shape[0]
        w = im.shape[1]
        lable_tree = ET.parse(self.label_dir_file)
        objs = lable_tree.findall('object')
        num = 0

        for obj in objs:
            if obj.find('name').text.lower().strip() != 'person':
                continue
            one_box = {}
            box = np.zeros(4)
            num = num + 1
            xml_bbox = obj.find('bndbox')
            x1 = max(min((float(xml_bbox.find('xmin').text) - 1)/w, 1), 0)
            y1 = max(min((float(xml_bbox.find('ymin').text) - 1)/h, 1), 0)
            x2 = max(min((float(xml_bbox.find('xmax').text) - 1)/w, 1), 0)
            y2 = max(min((float(xml_bbox.find('ymax').text) - 1)/h, 1), 0)
            box[:] = np.array([(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1])
            one_box['box'] = box
            if obj.find('difficult').text =='1':
                one_box['diff'] = True
            else:
                one_box['diff'] = False
                self.obj_num += 1
            one_box['def'] = False
            self.label_content.append(one_box)
        if num != 0:
            self.is_person = True

        if num!=len(self.label_content):
            print('error')
            exit()
        self.box_num = num


class voc_labels(object):
    def __init__(self):
        self.voc_dir = cfg.PASCAL_dir_test
        self.index_dir_file = os.path.join(self.voc_dir, 'ImageSets/Main/train.txt')
        self.image_dir = self.voc_dir + 'JPEGImages/'
        self.label_dir = self.voc_dir + 'Annotations/'
        self.rebuild = True
        self.labels = []
        self.total_obj_num = 0
        self.num_total_labels = 0
        self.image_size = cfg.IMAGE_SIZE
        self.prepare_li()
        self.count = 0
        self.finish = False

    def prepare_li(self):
        cache_file = os.path.join(cfg.CACHE_PATH, 'pascal_test_gt_labels.pkl')
        """
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Do not rebuilt, Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                self.labels = cPickle.load(f)
            self.num_total_labels = len(self.labels)
            print(self.num_total_labels)
            return
        """
        with open(self.index_dir_file, 'r') as f:
            image_index = [x.strip() for x in f.readlines()]
        for index in image_index:
            image_path = os.path.join(self.image_dir, index + '.jpg')
            label_path = os.path.join(self.label_dir, index + '.xml')
            label_image = label(image_path=image_path, label_path=label_path)
            if label_image.is_person:
                self.labels.append(label_image)
                self.total_obj_num += label_image.obj_num

        self.num_total_labels = len(self.labels)
        print(self.num_total_labels)
        return

    def get(self, batch_size):
        batch_labels = []
        n = 0
        for i in range(batch_size):
            batch_labels.append(self.labels[self.count])
            self.count+=1
            n += 1
            if not self.count < self.num_total_labels:
                self.count = 0
                self.finish = True
                break
        images = np.zeros((n,self.image_size, self.image_size, 3))
        for i in range(n):
            image = cv2.imread(batch_labels[i].image_dir_file)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image / 255.0) * 2.0 - 1.0
            images[i, :, :, :] = image
        return images, batch_labels





