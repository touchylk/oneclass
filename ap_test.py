# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import os
import cv2
import yolo.config as cfg
from yolo.yolo_net import YOLONet
import matplotlib.pyplot as plt
from utils.timer import Timer
import utils.ap_voc as ap_voc
import cPickle


class AP_test(object):

    def __init__(self):

        self.net = YOLONet(False)
        self.weights_file = cfg.WERIGHTS_READ
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.ap_iou_threshold = cfg.TEST_IOU_THRESHOLD
        self.nms_iou_threshold = cfg.NMS_IOU_THERSHOLD
        self.boundary2 = self.cell_size * self.cell_size * self.boxes_per_cell
        self.batch_size = cfg.TEST_batch_size
        self.total_obj = 0
        self.recall = None
        self.precise = None
        self.pro_num = 0

        """
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.variable_to_restore = tf.global_variables()
        print 'Restoring weights from: ' + self.weights_file
        saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        saver.restore(self.sess, self.weights_file)
        """


    def draw(self):
        self.test_ap()
        print(np.max(self.recall))
        print(np.max(self.precise))
        ap = 0
        for i in range(11):
            pi = self.precise[self.recall > (i / 10)]
            if pi.shape[0] == 0:
                p = 0
            else:
                p = np.max(pi)
            ap += p
        ap/=11
        print('recall is:{}'.format(np.max(self.recall)))
        print('all object is:{}'.format(self.total_obj))
        print('all proposel is :{}'.format(self.pro_num))
        print('AP is :{}'.format(ap))
        plt.figure()
        plt.plot(self.recall, self.precise)
        plt.show()

    def test_ap(self):
        result = self.run_dataset()
        argsort = np.argsort(-result[:,0])
        result[:, 0] = result[:, 0][argsort]
        result[:, 1] = result[:, 1][argsort]
        result[:, 2] = result[:, 2][argsort]
        pos = np.cumsum(result[:,1])
        neg = np.cumsum(result[:,2])
        print(self.pro_num)
        print(self.total_obj)
        self.recall = pos/self.total_obj
        self.precise = pos/(neg+pos)


    def run_dataset(self):
        """
        返回每一张图的
        :return:
        """
        rebult = True
        cache_file = os.path.join(cfg.CACHE_PATH, 'pascal_data.pkl')
        if rebult:
            data = ap_voc.voc_labels()
            print('saving labels to :' + cache_file)
            with open(cache_file, 'wb') as f:
                cPickle.dump(data, f)
        else:
            print('Do not rebuilt, Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                data = cPickle.load(f)
        self.total_obj = data.total_obj_num
        if self.total_obj==0:
            print('fsda error ashbc')
            exit()


        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        variable_to_restore = tf.global_variables()
        print 'Restoring weights from: ' + self.weights_file
        saver = tf.train.Saver(variable_to_restore, max_to_keep=None)
        saver.restore(sess, self.weights_file)

        result = np.zeros([data.num_total_labels*49,3])
        j = 0
        # 加入计时函数
        while not data.finish:
            """
            返回batch每一个图的正负样本及置信度,总目标数,
            """
            inputs, labels = data.get(self.batch_size)
            net_output = sess.run(self.net.logits, feed_dict={self.net.images: inputs})
            for i in range(len(labels)):
                image_result = self.nms_boxs(net_output[i])
                n = image_result.shape[0]
                label = labels[i]
                result[j:j+n, :] = self.pos_neg_simple(result_boxes=image_result, label=label)
                j += n
        sess.close()
        # 加入计时函数
        return result

    def pos_neg_simple(self,result_boxes,label):
        """
        result_boxes = np.zeros([scales_sorted.shape, 7])
         7中,0:3为box,4为执信度,5为正样本,6为负样本
        one_image.image_result为(n,5)
        :param one_image:
        :return:
        """
        num_result_box = result_boxes.shape[0]
        argsort = np.argsort(-result_boxes[:,4])

        result_boxes[:, 0] = result_boxes[:, 0][argsort]
        result_boxes[:, 1] = result_boxes[:, 1][argsort]
        result_boxes[:, 2] = result_boxes[:, 2][argsort]
        result_boxes[:, 3] = result_boxes[:, 3][argsort]
        result_boxes[:, 4] = result_boxes[:, 4][argsort]
        for i in range(num_result_box):
            iou_max = 0
            jmax = 0
            for j in range(label.box_num):
                bbnet = result_boxes[i,:4]
                bbgt = label.label_content[j]['box']
                iou = self.iou(bbnet,bbgt)
                if iou > iou_max:
                    iou_max =iou
                    jmax = j
            if iou_max > self.ap_iou_threshold:
                #print('OK')
                if not label.label_content[jmax]['diff']:
                    if not label.label_content[jmax]['def']:
                        result_boxes[i, 5] = 1
                        result_boxes[i, 6] = 0
                        label.label_content[jmax]['def'] = True
                    else:
                        result_boxes[i, 5] = 0
                        result_boxes[i, 6] = 1
            else:
                result_boxes[i, 5] = 0
                result_boxes[i, 6] = 1
        return result_boxes[:, 4:]

    def nms_boxs(self, net_output):
        """

        :param net_output:
        :return: result_boxes [n,7]
        0:3为box,4为置信度,5为正样本,7为负样本
        """

        boxes = np.reshape(net_output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        scales = np.reshape(net_output[0:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(boxes,[49,2,4])
        scales = np.reshape(scales,[49,2])
        per_cell_argsort = np.argsort(-scales)
        boxs_cell_filtered = np.zeros([49,4])
        scales_cell_filered = np.zeros([49])
        for i in range(49):
            if per_cell_argsort[i,0] == 0:
                boxs_cell_filtered[i,:] = boxes[i,0,:]
                scales_cell_filered[i] = scales[i, 0]
            else:
                boxs_cell_filtered[i, :] = boxes[i, 1, :]
                scales_cell_filered[i] = scales[i, 1]
        scales_argsort = np.argsort(-scales_cell_filered)
        scales_sorted = scales_cell_filered[scales_argsort]
        boxes_sorted = boxs_cell_filtered[scales_argsort]
        i=0
        while i< boxes_sorted.shape[0]:
            j = i+1
            while j < boxes_sorted.shape[0]:
                if self.iou(boxes_sorted[i], boxes_sorted[j]) > self.nms_iou_threshold:
                    boxes_sorted = np.delete(boxes_sorted,j ,axis=0)
                    scales_sorted = np.delete(scales_sorted, j)
                else:
                    j += 1
            i += 1
        #print(scales_sorted.shape)
        result_boxes = np.zeros([scales_sorted.shape[0], 7])
        self.pro_num += scales_sorted.shape[0]
        # 7中,0:3为box,4为执信度,5为正样本,6为负样本
        result_box = np.zeros(7)
        for i in range(scales_sorted.shape[0]):
            result_box[:4] = boxes_sorted[i]
            result_box[4] = scales_sorted[i]
            result_boxes[i, :] = result_box
        return result_boxes

    def iou(self,box1,box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)







def main():

    test = AP_test()
    test.draw()




if __name__ == '__main__':
    main()
