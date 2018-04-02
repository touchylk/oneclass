# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import datetime
import os
import yolo.config as cfg
#import matplotlib.pyplot as plt
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
slim = tf.contrib.slim



class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_to_restore = cfg.WEIGHTS_to_restore
        self.weights_to_save = cfg.WEIGHTS_to_save_dir
        self.max_iter = cfg.MAX_ITER
        self.train_process_save_txt = os.path.join(cfg.TRAIN_process_save_txt_dir,
                                                   '{}_yolo_verclass.txt'.format(datetime.datetime.now().strftime('%d_%H:%M')))
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.save_cfg()
        self.output_dir = cfg.OUTPUT_dir+'tf_summary'
        name_ = []
        self.num_layer = cfg.LAYER_to_restore
        if True:
            self.variable_to_restore = {}
            self.restore_name = slim.get_model_variables()
            i = 0
            for var in self.restore_name:
                #print(var)
                if i>96:
                    break
                if i< 24:
                    if var.op.name[12] =='w':
                        print(var.op.name[12:],i)
                        self.variable_to_restore[self.restore_name[i].op.name] = self.restore_name[i]
                elif i< 96:
                    if var.op.name[13] =='w':
                        print(var.op.name[13:],i)
                        self.variable_to_restore[self.restore_name[i].op.name] = self.restore_name[i]
                else:
                    if var.op.name[11] =='w':
                        print(var.op.name[11:],i)
                        self.variable_to_restore[self.restore_name[i].op.name] = self.restore_name[i]
                i +=1
            for i in self.variable_to_restore:

                name_.append(i)#self.variable_to_restore[self.restore_name[i].op.name])
            name_.sort()
            i = 0
            for vr in name_:
                print(vr,i)
                i += 1
            #exit()
        elif False:
            self.variable_to_restore = {}
            for i in range(self.num_layer):
                self.variable_to_restore[self.restore_name[i].op.name] = self.restore_name[i]
            for i in range(len(self.variable_to_restore)):
                print('{}\n{}'.format(self.restore_name[i].op.name,self.variable_to_restore[self.restore_name[i].op.name]))
                with open(self.train_process_save_txt, 'a') as f:
                    f.writelines('{}\n{}\n'.format(self.restore_name[i].op.name,self.variable_to_restore[self.restore_name[i].op.name]))

        else:
            self.variable_to_restore = tf.global_variables()
        self.variable_to_save = tf.global_variables()

        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(self.variable_to_save, max_to_keep=None)
        print('pass')
        self.ckpt_file = os.path.join(self.weights_to_save, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session()#config=config)
        self.sess.run(tf.global_variables_initializer())

        if 0:#self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_to_restore)
            with open(self.train_process_save_txt, 'a') as f:
                f.writelines('Restoring weights from: ' + self.weights_to_restore+'\n')


            self.restorer.restore(self.sess, self.weights_to_restore)
        else:
            print('does not init!!!')
            with open(self.train_process_save_txt, 'a') as f:
                f.writelines('does not init!!!\n')



        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()
        sum_loss = np.zeros([cfg.MAX_ITER+1],dtype=float)
        #plt.axis([0, cfg.MAX_ITER, cfg.AX_LOW, cfg.AX_HIGHT])
        #plt.ion()

        for step in xrange(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images, self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    sum_loss[step] = loss
                    #plt.scatter(step,loss)
                    #plt.pause(0.1)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
                        ' Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.epoch,
                        int(step)+cfg.LAST_STEP,
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)
                    with open(self.train_process_save_txt,'a') as f:
                        f.writelines(log_str+'\n')

                else:
                    train_timer.tic()
                    summary_str,loss= self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    sum_loss[step] = loss

                    train_timer.toc()
                    #print('\nb')
                    #print(summary_str)

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                loss= self.sess.run(self.train_op, feed_dict=feed_dict)
                sum_loss[step] = loss

                train_timer.toc()
                #print('q')

            if step % self.save_iter == 0:
                print('Saving checkpoint file to:{}-{}'.format(self.ckpt_file,step+cfg.LAST_STEP))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=self.global_step+cfg.LAST_STEP)
                with open(self.train_process_save_txt, 'a') as f:
                    f.writelines('Saving checkpoint file to:{}-{}\n'.format(self.ckpt_file,step+cfg.LAST_STEP))


    def save_cfg(self):

        with open(os.path.join('data', 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)




#    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():


    if not os.path.exists(cfg.OUTPUT_dir):
        os.makedirs(cfg.OUTPUT_dir)
    if not os.path.exists(cfg.TRAIN_process_save_txt_dir):
        os.makedirs(cfg.TRAIN_process_save_txt_dir)
    if not os.path.exists(cfg.WEIGHTS_output_dir):
        os.makedirs(cfg.WEIGHTS_output_dir)




#    if args.data_dir != cfg.DATA_PATH:
#        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train',rebuild=False)

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
