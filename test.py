# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE  # 448
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.C_THRESHOLD
        self.iou_threshold = cfg.NMS_IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.variable_to_restore = tf.global_variables()

        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][0])
            y = int(result[i][1])
            w = int(result[i][2] / 2)
            h = int(result[i][3] / 2)
            print x,y,w,h
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, ':%.2f' % result[i][4], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(img, result[i][0] + ':%.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            #print '{} P:{:.2f} P_Class:{:.2f} Pobj*IOU:{:.2f}'.format(result[i][0],result[i][5],result[i][6],result[i][7])
            #print result[i][5]

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]
        print('{},fff'.format(len(result)))

        for i in range(len(result)):
            result[i][0] *= (1.0 * img_w / self.image_size)
            result[i][1] *= (1.0 * img_h / self.image_size)
            result[i][2] *= (1.0 * img_w / self.image_size)
            result[i][3] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        """
        输入图片,通过self.interpret_output输出检测结果.
        :param inputs:
        :return results:
        """
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))
        return results

    def interpret_output(self, output):
        """

        :param output:

        predict_scales = tf.reshape(predicts[:, :self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            # predict_scales为(batch_size,7,7,2)
        predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        :return:

        result list,单元为:
        [self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]]
        """



        # 将net输出的坐标,reshape成(7,7,2,4)维,boxes
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        # 到这里,boxes为(7,7,2,4),(:,:,:,0:1)为0-1之间,表示整张图的相对位置
        # (:,:,:,2：)为平方后的值,即直接就是长和宽
        boxes *= self.image_size
        # 到这里,boxes为(7,7,2,4),整张图的绝对位置,0-448
        scales = np.reshape(output[0:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        # 将net输出的IOU,reshape成(7,7,2)维
        threshold_filer_0 = np.array(scales >= self.threshold,dtype='bool')
        threshold_filer_1 = np.nonzero(threshold_filer_0)
        scales_thres_filed = scales[threshold_filer_0]
        # scales_thres_filed为（n）维
        boxes_thres_filed = boxes[threshold_filer_1[0],threshold_filer_1[1],threshold_filer_1[2]]
        # boxes_thres_filed为(n,4)

        sort_s = np.array(np.argsort(scales_thres_filed))[::-1]
        scales_thres_filed_sorted = scales_thres_filed[sort_s]
        boxes_thres_filed_sorted = boxes_thres_filed[sort_s]
        iou_filter = np.ones_like(scales_thres_filed_sorted,dtype='bool')
        for i in range(scales_thres_filed_sorted.shape[0]):
            if scales_thres_filed_sorted[i] == 0:
                continue
            for j in range(i+1,scales_thres_filed_sorted.shape[0]):
                if self.iou(boxes_thres_filed_sorted[i],boxes_thres_filed_sorted[j])>self.iou_threshold:
                    iou_filter[j] = False
                    scales_thres_filed_sorted[j] = 0

        scales_thres_filed_sorted_ioufiled = scales_thres_filed_sorted[iou_filter]
        boxes_thres_filed_sorted_ioufiled = boxes_thres_filed_sorted[iou_filter]

        result = []
        for i in range(len(scales_thres_filed_sorted_ioufiled)):
            result.append([boxes_thres_filed_sorted_ioufiled[i][0],boxes_thres_filed_sorted_ioufiled[i][1],
                           boxes_thres_filed_sorted_ioufiled[i][2],boxes_thres_filed_sorted_ioufiled[i][3],
                          scales_thres_filed_sorted_ioufiled[i]])
        return result


    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        #output='output'.format(imname)
        #print output
        #cv2.imwrite('test113.jpg',image)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = ''#cfg.GPU

    yolo = YOLONet(False)

    weight_file = cfg.WERIGHTS_READ #os.path.join(cfg.WEIGHT_READ,'save.ckpt-{}'.format(cfg.LAST_STEP))
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = cfg.IMAGE_dir_file
    detector.image_detector(imname)


if __name__ == '__main__':
    main()
