import tensorflow as tf
import os
from SRGAN import *

flags = tf.app.flags
flags.DEFINE_float('lr', 0.0001, 'learning rate')                                                           # 学习率
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.999, 'beta2')
flags.DEFINE_float('lambd', 0.001, 'coeff for adversarial loss')
flags.DEFINE_string('dataset_dir', 'data', 'dataset directory')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('sample_dir', 'sample', 'sample directory')
flags.DEFINE_string('test_dir', 'test', 'test directory')
flags.DEFINE_string('model_dir', 'ImageNet', 'using imagenet dataset')
flags.DEFINE_string('logs_dir', 'logs', 'log directory')
flags.DEFINE_bool('is_crop', True, 'crop images')
flags.DEFINE_integer('epoches', 200, 'training epoches')
flags.DEFINE_integer('fine_size', 256, 'fine size')
flags.DEFINE_string('train_set', 'ImageNet', 'train phase')
flags.DEFINE_string('val_set', 'Set5', 'val phase')
flags.DEFINE_string('test_set', 'Set14', 'test phase')
flags.DEFINE_bool('is_testing', False, 'training or testing')
flags.DEFINE_bool('is_training', False, 'training or testing')
# 使用flags定义命令行参数的方法
FLAGS = flags.FLAGS

# 检测是否存在必须的文件夹，不存在便创建
def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.mkdir(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.mkdir(FLAGS.logs_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.mkdir(FLAGS.test_dir)


def main(_):
    check_dir()
    # 指定GPU进程中使用显存的上限，allow_growth=True：刚一开始分配少量的GPU容量然后按需慢慢的增加
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    # 配置Session运行参数和GPU设备指定，allow_soft_placement=True：如果你指定的设备不存在允许TF自动分配设备
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    
    with tf.Session(config=config) as sess:
        # def __init__(self, config, batch_size=1, input_height=256, input_width=256, input_channels=3, sess=None):
        srgan = SRGAN(FLAGS, batch_size=8, input_height=256, input_width=256, input_channels=3, sess=sess)
        srgan.build_model()
        
        print ("开始训练")
        srgan.train()
        #srgan.test()
        print ("结束训练")
        
if __name__=='__main__':
    # 指定Session在第1块GPU上运行：
    with tf.device('/gpu:0'):
        tf.app.run()
        
        
        
        
        
        
        
        
        
