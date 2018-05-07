# coding='utf-8'
'''
    An re-implementation of MobileNet_V2
    author: Youzhao Yang
    date: 05/01/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import os

from MobileNet_V2 import MobileNet_V2

flags = tf.app.flags
flags.DEFINE_integer('input_height', 224, 'input image height')
flags.DEFINE_integer('input_width', 224, 'input image width')
flags.DEFINE_integer('input_channel', 3, 'input image channel')
flags.DEFINE_integer('batchsize', 128, 'training batchsize')
flags.DEFINE_integer('iterations', 400000, 'training epoches')
flags.DEFINE_integer('num_class', 101, 'numbers of classes')
flags.DEFINE_float('learning_rate', 0.0001, 'training rate')
flags.DEFINE_float('learning_rate_decay', 0.98, 'learning rate decay')
flags.DEFINE_float('decay', 0.9, 'decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_float('weight_decay', 5e-4, 'weigh decay')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('logs_dir', 'logs', 'logs directory')
flags.DEFINE_string('dataset', 'Celtech101', 'name of dataset')
flags.DEFINE_bool('is_training', False, 'training phase or not')
flags.DEFINE_bool('is_testing', False, 'testing phase or not')

FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.mkdir(FLAGS.logs_dir)

def print_config():
    print('Config Proto:')
    print('-'*30)
    print('dataset:{}'.format(FLAGS.dataset))
    print('input_height:{}'.format(FLAGS.input_height))
    print('input_width:{}'.format(FLAGS.input_width))
    print('input_channel:{}'.format(FLAGS.input_channel))
    print('batchsize:{}'.format(FLAGS.batchsize))
    print('iterations:{}'.format(FLAGS.iterations))
    print('learning_rate:{}'.format(FLAGS.learning_rate))
    print('learning_rate_decay:{}'.FLAGS.learning_rate_decay)
    print('decay:{}'.format(FLAGS.decay))
    print('momentum:{}'.format(FLAGS.momentum))
    print('weight_decay:{}'.format(FLAGS.weight_decay))
    print('is_training:{}'.format(FLAGS.is_training))
    print('-'*30)

def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    gpu_options.allow_growth = True
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
    
    with tf.Session(config=config) as sess:
        mobilenet_v2 = MobileNet_V2(config=FLAGS, sess=sess)
        mobilenet_v2.build_model()
        if FLAGS.is_training:
            mobilenet_v2.train_model()
        if FLAGS.is_testing:
            mobilenet_v2.test_model()

if __name__=='__main__':
    tf.app.run()
