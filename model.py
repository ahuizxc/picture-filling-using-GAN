import tensorflow as tf
from ops import fully_connect
from ops import conv2d
from ops import pooling
from ops import transconv2d
from ops import batch_norm
from ops import lrelu
import numpy as np


def generator(masked_image, batch_size, image_dim, is_train=True, no_reuse=False):
  with tf.variable_scope('generator') as scope:
    if not (is_train or no_reuse):
      scope.reuse_variables()
      
    # input 180x256ximage_dim
    #conv0_1
    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(masked_image, 32, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #conv0_2
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 32, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 180*256*32 
    
    #pool0
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = pooling(hidden, (2, 2), (2,2))
    # output 90*128*32
    
    #conv1_1_new
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 64, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv1_2
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 64, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #pool1
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = pooling(hidden, (2, 2), (2,2))
    #output 45*64*64
    
    #conv2_1
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 128, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #conv2_2
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 128, (3, 3), (3, 2), trainable=is_train)                 ###
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #pool2
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = pooling(hidden, (2, 2), (1,1))                            ###
    #output 15*32*128
    
    #conv3_1
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 256, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #conv3_2
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 256, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #conv3_3
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 256, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #conv3_4
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 512, (3, 3), (3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))   
    
    #pool3
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = pooling(hidden, (2, 2), (1,2))
    #output 5*8*512
    
    #fc3
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = tf.reshape(hidden, [batch_size, 5 * 8 * 512])
      hidden = fully_connect(hidden, 5120, trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 5120
    
    #defc3
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = fully_connect(hidden, 20480, trainable=is_train)
      hidden = tf.reshape(hidden, [batch_size, 5, 8, 512])
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 5*8*512
    
    
    #conv_decode3_4
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=512,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 5*8*512
     
    #conv_decode3_3  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=512,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    
    #conv_decode3_2  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=512,
                           kernel=(3, 3), stride=(3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 15*16*512
  
    #conv_decode3_1  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=512,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #conv_decode2_2  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=256,
                           kernel=(3, 3), stride=(3, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 45*16*256
    
    #conv_decode2_1  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=256,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #conv_decode1_2  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=128,
                           kernel=(3, 3), stride=(1, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 45*32*128 
     
    #conv_decode1_1  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=128,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #conv_decode0_2  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=64,
                           kernel=(3, 3), stride=(2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 90*64*64
    
    #conv_decode0_1  
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=64,
                           kernel=(3, 3), stride=(1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #reconstruction
    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=1,
                           kernel=(3, 3), stride=(2, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #output 180*64*image_dim
    
    return hidden
    

def global_discriminator(full_image, batch_size, reuse=False, is_train=True):
  with tf.variable_scope('global_discriminator') as scope:
    if reuse:
      scope.reuse_variables()
      
    #input 180*256*1
    #conv0_1
    layer_num = 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(full_image, 32, (3, 3), (2, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 90*256*32  
      
    #conv0_2
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 32, (4, 4), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 90*256*32
    
    #conv1_1
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 64, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv1_2
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 64, (4, 4), (1, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 90*128*64
      
    #conv2_1
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 128, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv2_2
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 128, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 45*64*128
      
    #conv3_1
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 256, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv3_2
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 256, (4, 4), (3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 15*32*256
      
    #conv4_1
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 512, (3, 3), (1, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv4_2
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = conv2d(hidden, 512, (4, 4), (3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 5*8*512  
    
    # #conv5
    # layer_num += 1
    # with tf.variable_scope('hidden' + str(layer_num)):
      # hidden = conv2d(hidden, 1, (4, 4), (1, 1), trainable=is_train)
      # hidden = lrelu(batch_norm(hidden, train=is_train))
    # #
    layer_num += 1
    with tf.variable_scope('hidden_glo' + str(layer_num)):
      hidden = tf.reshape(hidden, [batch_size, 5 * 8 * 512])
      hidden = fully_connect(hidden, 1, trainable=is_train)
      #输出尺寸为true 或者false
    return hidden[:, 0]

def local_discriminator(fake_image, batch_size, reuse=False, is_train=True):
  with tf.variable_scope('local_discriminator1') as scope:
    if reuse:
      scope.reuse_variables()

    #input 180*64*image_dim
    #conv1_1
    layer_num = 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(fake_image, 64, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv1_2
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 64, (4, 4), (2, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 90*64*64
    
    #conv2_1
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 128, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    
    #conv2_2
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 128, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 45*32*128
    
    #conv3_1
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 256, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 45*32*256  

    #conv3_2
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 256, (4, 4), (3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 15*16*256
    
    #conv4_1
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 512, (3, 3), (1, 1), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
      
    #conv4_2
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = conv2d(hidden, 512, (4, 4), (3, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    #output 5*8*512
      
    # #conv5
    # layer_num += 1
    # with tf.variable_scope('hidden' + str(layer_num)):
      # hidden = conv2d(hidden, 1, (4, 4), (1, 1), trainable=is_train)
      # hidden = lrelu(batch_norm(hidden, train=is_train))
    # #
    layer_num += 1
    with tf.variable_scope('hidden_loc' + str(layer_num)):
      hidden = tf.reshape(hidden, [batch_size, 5* 8 * 512])
      hidden = fully_connect(hidden, 1, trainable=is_train)
      #输出尺寸为true 或者false
    return hidden[:, 0]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
