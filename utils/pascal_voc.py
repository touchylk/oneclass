# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy
import yolo.config as cfg

"""第134行,要注意,是否修改,当图片中没有所要目标时,还应当进行训练"""

class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = cfg.PASCAL_PATH
        self.data_path = os.path.join(self.devkil_path, 'VOC2012')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.epoch_changed = False
        self.gt_labels = None
        # 下面执行了prepare,已经给gt_labels赋值了
        self.prepare()

    def get(self):
        """
        重要!!!


        记录训练过程,用文件保存.

        labels:
        为(batch_size,7,7,5),7,7为格子,目标储存在对应的格子的维数上面
        5为目标标签,0为1,1:3为中心点,取值在,1:3为中心点,取值在0-447之间,相对于整张图,3:5为长和宽,0-447,需要翻转的已经翻转.

        images:
        为(batch_size,448,448,3)与标签相对应

        返回值为网络的输入值

        """
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 5))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
                self.epoch_changed = True
            else:
                self.epoch_changed = False


        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels() # 这里从VOC中读取label
        # 下面是进行翻转的,在get()中,图像对应的是左右翻转,所以对应的标签应该左右翻转,
        # 由于label储存的时候是安装格子的对应顺序储存的,所以先将label(7,7,5)的中间一维翻转,
        # 再将中心点横坐标翻转(即减去总和).
        # 将获取的gt_labels打乱了,文件中不变
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        """
        返回gt_labels,是一个list,长为索引文件的长度,每一个单元是{'imname': imname, 'label': label, 'flipped': False}
        并将gt_labels结构化储存在文件.
        imname为图片的位置.
         label为(7,7,5)表示一个图上面的种类标签,7,7为格子,5中0为1,
        1:5为中心店和长和宽的坐标,1:3为中心点,取值在0-447之间,相对于整张图,3:5为长和宽,0-447
        :return:
        """
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        # 这个if语句判断了是否需要重新从VOC中读取标签,
        # 如果不需要,就直接从之前已经结构化保存的list类里边读取.cache_file
        if os.path.isfile(cache_file) and not self.rebuild:
            print('doesnot rebuilt Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = cPickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        # 如果文件不存在,则创建文件:cache_file
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # 这里读取VOC中训练或者测试的图片索引,储存在self.image_index中
        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main',
                                   'train.txt')

        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main',
                                   'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
            print('index num is {}'.format(len(self.image_index)))
            #print('{},{}'.format(self.image_index[2],'image_index'))
#        self.image_index = [x.strip() for x in range(1,1000)]

        # 按索引读取标签gt_labels
        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            #print('{},{}'.format(index,num))
            # print('{},{}'.format(num,'num'))
            if num == 0:
                if np.random.binomial(1, 0.9, 1):
                    #pass
                    continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        print('the num of image is {}'.format(len(gt_labels)))
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        返回label和num
        label为(7,7,5)表示一个图上面的种类标签,7,7为格子,5中0为1,
        1:5为中心店和长和宽的坐标,1:3为中心点,取值在0-447之间,相对于整张图,3:5为长和宽,0-447
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')

        im = cv2.imread(imname)
        #print('{},{}'.format(im.shape,'imshape'))
        #print imname
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 5))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num = 0

        for obj in objs:
            itt=0
            iname=obj.find('name')

            if obj.find('name').text.lower().strip()!='person':
                continue

            num = num+1
                
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            # 这里,x1,y1,x2,y2是左上点和右下店的坐标,取值在0到447之间
            #cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            #print cls_ind
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            #label[y_ind, x_ind, 5 + cls_ind] = 1
        # label[cell_size,cell_size,5], 5:{0为1, 1-4为
        return label, num # len(objs)

