# coding = 'utf-8'
'''
    An re-implementation of MobileNet_V2
    author: Youzhao Yang
    date: 05/01/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import numpy as np

from tqdm import tqdm
from utils import *

class MobileNet_V2:
    '''
        MobileNet_V2 class
    '''
    model_name = 'MobileNet_V2.cpkt'
    
    def __init__(self, config=None, sess=None):
        # datasets params
        self.input_height = config.input_height
        self.input_width = config.input_width
        self.input_channel = config.input_channel
        self.num_class = config.num_class
        
        # training params
        self.batchsize = config.batchsize
        self.width_multiplier = 6
        
        # configuration
        self.config = config
        self.sess = sess

    def inverted_bottleneck_block(self, input_x, channel_up_factor, output_channel, subsample, is_training=True):
        self.num_block = self.num_block + 1
        scope='inverted_bottleneck{}_{}_{}'.format(self.num_block, channel_up_factor, subsample)
        
        with tf.variable_scope(scope) as scope:
            # set stride
            stride = 2 if subsample else 1
            # get numbers of input channel
            input_channel = input_x.get_shape().as_list()[-1]
            
            channel_up_ops = slim.conv2d(input_x, channel_up_factor*input_channel, 1, 1)
            separable_ops = slim.separable_conv2d(channel_up_ops, None, 3, depth_multiplier=1, stride=stride)
            depth_wise_ops = slim.conv2d(separable_ops, output_channel, 1, 1, activation_fn=None)
            
            # residual add if input_channel == output_channel
            is_conv_res = False if depth_wise_ops.get_shape().as_list()[-1] == input_channel else True
            is_residual = True if depth_wise_ops.get_shape().as_list()[-1] == input_channel else False
            
            # normal residual
            if is_residual:
                output = input_x + depth_wise_ops
                
            # if the numbers of channel of the input and output don't matching, conv2d_1*1 is required
            elif stride == 1 and is_conv_res:
                output_channel = depth_wise_ops.get_shape().as_list()[-1]
                output = slim.conv2d(input_x, output_channel, 1, 1, activation_fn=None) + depth_wise_ops
                
            else:
                output = depth_wise_ops

            return output

    def mobilenet_v2(self, input_x, is_training=True, reuse=False, keep_prob=0.8, scope='mobilenet_v2'):
        # batch_norm parameters
        # bn_parameters = {'is_training': is_training, 'center':True, 'scale':True, 'decay':0.997}
        
        self.num_block = 0
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()

            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                                      normalizer_fn=slim.batch_norm,
                                                                      #normalizer_params=bn_parameters,
                                                                      activation_fn=tf.nn.relu6),\
                    slim.arg_scope([slim.dropout], keep_prob=keep_prob) as s:

                conv0 = slim.conv2d(input_x, 32, 3, stride=1, scope='conv0')
                
                # bottleneck_residual_block
                bottleneck_1_1 = self.inverted_bottleneck_block(conv0, 1, 16, False, is_training=is_training)
                bottleneck_2_1 = self.inverted_bottleneck_block(bottleneck_1_1, self.width_multiplier, 24, False, is_training=is_training)
                bottleneck_2_2 = self.inverted_bottleneck_block(bottleneck_2_1, self.width_multiplier, 24, False, is_training=is_training)
                bottleneck_3_1 = self.inverted_bottleneck_block(bottleneck_2_2, self.width_multiplier, 32, True, is_training=is_training)
                bottleneck_3_2 = self.inverted_bottleneck_block(bottleneck_3_1, self.width_multiplier, 32, False, is_training=is_training)
                bottleneck_3_3 = self.inverted_bottleneck_block(bottleneck_3_2, self.width_multiplier, 32, False, is_training=is_training)
                bottleneck_4_1 = self.inverted_bottleneck_block(bottleneck_3_3, self.width_multiplier, 64, True, is_training=is_training)
                bottleneck_4_2 = self.inverted_bottleneck_block(bottleneck_4_1, self.width_multiplier, 64, False, is_training=is_training)
                bottleneck_4_3 = self.inverted_bottleneck_block(bottleneck_4_2, self.width_multiplier, 64, False, is_training=is_training)
                bottleneck_4_4 = self.inverted_bottleneck_block(bottleneck_4_3, self.width_multiplier, 64, False, is_training=is_training)
                bottleneck_5_1 = self.inverted_bottleneck_block(bottleneck_4_4, self.width_multiplier, 96, False, is_training=is_training)
                bottleneck_5_2 = self.inverted_bottleneck_block(bottleneck_5_1, self.width_multiplier, 96, False, is_training=is_training)
                bottleneck_5_3 = self.inverted_bottleneck_block(bottleneck_5_2, self.width_multiplier, 96, False, is_training=is_training)
                bottleneck_6_1 = self.inverted_bottleneck_block(bottleneck_5_3, self.width_multiplier, 160, True, is_training=is_training)
                bottleneck_6_2 = self.inverted_bottleneck_block(bottleneck_6_1, self.width_multiplier, 160, False, is_training=is_training)
                bottleneck_6_3 = self.inverted_bottleneck_block(bottleneck_6_2, self.width_multiplier, 160, False, is_training=is_training)
                bottleneck_7_1 = self.inverted_bottleneck_block(bottleneck_6_3, self.width_multiplier, 320, False, is_training=is_training)
                
                conv8 = slim.conv2d(bottleneck_7_1, 1280, 3, stride=1, scope='conv8')
                
                # global average pooling
                filter_size = [conv8.get_shape().as_list()[1], conv8.get_shape().as_list()[2]]
                avgpool = slim.avg_pool2d(conv8, filter_size, scope='avgpool')
                dropout = slim.dropout(avgpool)
                
                logits = tf.squeeze(slim.conv2d(dropout, self.num_class, 1, stride=1, activation_fn=None,
                                                                                       normalizer_fn=None))
            return logits
            
    def build_model(self):
        # classicification task
        self.input_x = tf.placeholder(tf.float32, shape=[self.batchsize, self.input_height, self.input_width, self.input_channel], name='input_x')
        self.input_label = tf.placeholder(tf.float32, shape=[self.batchsize, self.num_class], name='input_label')
        
        # learning_rate
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
        
        # mobilenet_v2 forward
        self.train_logits = self.mobilenet_v2(self.input_x, is_training=True, reuse=False, keep_prob=0.5)
        
        # predicted softmax
        self.test_logits = self.mobilenet_v2(self.input_x, is_training=True, reuse=True, keep_prob=1)
        self.pred_softmax = tf.nn.softmax(self.test_logits, 1)

        # loss function
        def softmax_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.softmax_cross_entropy_with_logits(targets=x, labels=y)
        
        # weight regularization
        self.weights_regularization = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss with weight regularization
        self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.train_logits, self.input_label)) + self.config.weight_decay*self.weights_regularization
        
        # optimizer
        '''
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                               beta1 = 0.9,
                                               beta2 = 0.999).minimize(self.loss)
        '''
        self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                               decay=self.config.decay,
                                               momentum=self.config.momentum).minimize(self.loss)
        
        # summary
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.logs_dir, self.sess.graph)
        
        # saver
        self.saver = tf.train.Saver()
        
    def train_model(self):
        # initialize variables
        tf.global_variables_initializer().run()
        
        # load model
        if self.load_model():
            print('[!!!] load model successfully')
        else:
            print('[***] fail to load model')
                
        # learning rate
        learning_rate = self.config.learning_rate
        
        # get data
        datasource = get_data(self.config.dataset, is_training=True)
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=True)
        ites_per_epoch = int(len(datasource.images)/self.batchsize)
        
        for ite in tqdm(range(self.config.iterations)):
            images, labels = next(gen_data)
            _, loss, summaries = self.sess.run([self.optim,
                                                self.loss, 
                                                self.summaries], feed_dict={self.input_x:images,
                                                                             self.input_label:labels, 
                                                                             self.learning_rate:learning_rate})
            
            self.summary_writer.add_summary(summaries, global_step=ite)
            
            if np.mod(ite, ites_per_epoch) == 0:
                # learning rate decay
                # learning_rate = learning_rate*self.config.learning_rate_decay
                self.test_model(int(ite/ites_per_epoch))
            
            if np.mod(ite, 10*ites_per_epoch)==0:
                self.save_model()

    @property
    def model_dir(self):
        return '{}/{}'.format(self.config.checkpoint_dir, self.config.dataset)
    
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.config.dataset, self.model_name)
    
    def save_model(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        self.saver.save(self.sess, self.model_pos)
        
    def load_model(self):
        if not os.path.isfile(os.path.join(self.model_dir,'checkpoint')):
            return False
        else:
            self.saver.restore(self.sess, self.model_pos)
            return True

    def test_model(self, epochs):        
        # initialize variables
        if not self.config.is_training:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            
            if self.load_model():
                print('[!!!] load model successfully')
            else:
                print('[***] fail to load model')
        
        datasource = get_data(self.config.dataset, is_training=False)        
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=False)
        
        ites = int(len(datasource.images)/self.batchsize)
        correct_num = 0
        for ite in range(ites):
            images, labels = next(gen_data)
            pred_softmax = self.sess.run([self.pred_softmax], feed_dict={self.input_x:images})
            correct = np.sum(np.equal(np.argmax(labels, 1), np.argmax(pred_softmax[0], 1)).astype(np.float32))
            correct_num = correct_num + correct
        
        correct_rate = float(correct_num)/float(ites*self.batchsize)
        print('test-epochs {}: -- accuracy:{:.4f} --'.format(epochs, correct_rate))

if __name__=='__main__':
    input_x = tf.Variable(tf.random_normal([64,224,224,3]), dtype=tf.float32, name='input')
    mobilenet_v2 = MobileNet_V2()
    start_time = time.time()
    output = mobilenet_v2.mobilenet_v2(input_x)
    end_time = time.time()
    print('total time:{:.4f}'.format(end_time-start_time))
