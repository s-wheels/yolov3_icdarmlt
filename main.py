#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:28:53 2018
@author:  Sean Wheeler

This program builds a parser and uses it to determine hyperparameters for the YOLO network
Takes input folders for images and ground truth
Builds YOLO network and runs it in tensorflow
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import time
import yolo_net

## TRAINING IS BEST USING NCHW, INFERENCE/PREDICTION IS BEST USING NHWC
data_format = 'NCHW'
num_images = 7200
split = False     #Which set of anchors to use
trn_img_dir = 'ICDAR2017_MLT/processed_data/training_images'
trn_lab_dir = 'ICDAR2017_MLT/processed_data/training_labels_onehot'
val_img_dir = 'ICDAR2017_MLT/processed_data/validation_images'
val_lab_dir = 'ICDAR2017_MLT/processed_data/validation_labels_onehot'

if split == False: #Use the anchors from Kmeans on entire training dataset
    anchors = [(6,5),(18,10),(37,13),(25,41),(63,23),(105,33),(67,92),(173,57),(110,234),(296,95)]
else:              #Use the anchors from the split dataset
    anchors = [(5,3),(7,9),(14,4),(45,11),(14,22),(23,10),(120,40),(253,84),(88,170),(54,35)]
    

## set hyperparams

batch_norm_decay = 0.9
batch_norm_epsilon = 1e-05
leaky_relu = 0.1
num_epochs = 150
batch_size = 8
num_scales = 3
num_anchors = 3

#% Reset tensorflow flags, sessions and graph


FLAGS = tf.app.flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

tf.Session().close()
tf.reset_default_graph()


## MODEL INPUT PARAMETERS
tf.app.flags.DEFINE_integer('img_width', 416, 'Image width')
tf.app.flags.DEFINE_integer('img_height', 416, 'Image height')
tf.app.flags.DEFINE_integer('img_channels', 3, 'Image channels')
tf.app.flags.DEFINE_string('class_names', 'icdar_mlt.names', 'File with class names')
tf.app.flags.DEFINE_integer('num_classes', 9, 'Number of classes  ')


## PARAMETERS
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint.')
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'Number of steps between logging results to the console and saving summaries')
tf.app.flags.DEFINE_integer('save_model', 1, 'Number of steps between model saves')
tf.app.flags.DEFINE_string('ckpt_file', 'saved_icdarmlt_model/model.ckpt', 'Where to save checkpoint models')
tf.app.flags.DEFINE_string('pretrained_file', 'saved_darknet_model/model.ckpt', 'Pre-trained Darknet model')


## HYPERPARAMETERS
tf.app.flags.DEFINE_integer('num_epochs', num_epochs,
                            'Number of epochs to train for.  ')
tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Number of examples per mini-batch  ')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'Decay the learning rate every 1000 steps')
tf.app.flags.DEFINE_float('decay_rate', 0.8, 'The base of our exponential for the decay')


## HARDWARE PARAMETERS
tf.app.flags.DEFINE_float('gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')
tf.app.flags.DEFINE_string('data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')


# Main Function



    
def yolo():
    
    
    #Configure GPU Options
    
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction),
        log_device_placement=False,
    )
    
    
    #BUILD TF GRAPH

    images_ph = tf.placeholder(tf.float32, [batch_size, FLAGS.img_height, FLAGS.img_width, 3])
    labels_ph = tf.placeholder(tf.float32, [batch_size, None, 4+FLAGS.num_classes])
    labels_gr_ph = tf.placeholder(tf.int32, [batch_size, None,num_scales*num_anchors])


    with tf.variable_scope('detector'):
        predictions = yolo_net.yolo_v3(images_ph, FLAGS.num_classes, anchors, is_training=True)
        
        
    labels_assigned, obj_present = yolo_net.tf_assign_label(labels_ph, labels_gr_ph, predictions)
    cost = yolo_net.yolo_cost(labels_assigned, obj_present, predictions, labels_ph, batch_size)
    
    global_step = tf.Variable(0, trainable=False)
    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                               FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_learning_rate).minimize(cost, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(scope='detector'), max_to_keep=10)

    #EXECUTE GRAPH - FEED IN: IMAGE, INPUT_LABELS, LABELS_GRIDS
    
     
    with tf.Session(config = config) as sess:
        
        sess.run(tf.global_variables_initializer())
        batch_range = int(num_images/batch_size)

        for epoch in range(num_epochs):
            t_st = time.time()
            epoch_cost = 0
            
            for batch_num in range(batch_range):
                batch_st = 1+(batch_num*batch_size)
                img_range = range(batch_st, batch_st+batch_size)
                images = load_images_fd(img_range, trn_img_dir, tensor=False)
                labels = load_labels_fd(img_range, trn_lab_dir)
                labels_gr = assign_grid_box(labels)
                
                #RUN SESSION
                _ , batch_cost = sess.run([optimizer, cost], feed_dict={images_ph: images,
                                          labels_ph: labels, labels_gr_ph: labels_gr}) 
                epoch_cost += batch_cost
                
                if (batch_num % 20==0):
                    print(batch_num, batch_cost)


            print('Epoch {0} trained in {1:.2f}s'.format(epoch, time.time()-t_st))
            print('Epoch {0} cost {1}'.format(epoch, epoch_cost))

            if (epoch % FLAGS.save_model == 0):
                print('Saving model')
                saver.save(sess, save_path=FLAGS.ckpt_file, global_step=epoch)
                    
    return out






#%% Loading Functions
    
    
def load_labels_fd(labels_range, labels_dir, num_classes=9):
    """
    ARGS:
        labels_range = int or range of labels to be read in
        label_dir = directory where labels are stored
        
    OUTPUTS:
        labels = np array of shape (batch, num_labels, 5)
        
    """
    if type(labels_range)==int:
        labels_file = labels_dir + "/label_" + str(labels_range) + ".txt"
        labels = np.loadtxt(labels_file, dtype = np.float32, delimiter=',')
    elif type(labels_range)==range:
        labels_file = labels_dir + "/label_" + str(labels_range[0]) + ".txt"
        labels = np.loadtxt(labels_file, dtype = np.float32, delimiter=',')
        
        lab_len = len(labels)
        
        if len(labels.shape)==1:
            lab_len=1
            labels = np.expand_dims(labels, axis=0)
        
        labels = np.expand_dims(labels, axis=0)
            
        for i in range(labels_range[0]+1,labels_range[-1]+1):
            labels_file = labels_dir + "/label_" + str(i) + ".txt"
            labels_int = np.loadtxt(labels_file, dtype = np.float32, delimiter=',')
            len_lab_int = len(labels_int) if len(labels_int.shape)==2 else 1
            if len_lab_int==1:
                labels_int = np.expand_dims(labels_int, axis=0) 
            chg_len = len_lab_int - lab_len
            if chg_len<0:
                labels_int = np.concatenate((labels_int, np.zeros((abs(chg_len),4+num_classes), dtype=np.float32)), axis=0)
            elif chg_len>0:
                labels = np.concatenate((labels, np.zeros((labels.shape[0],chg_len,4+num_classes), dtype=np.float32)), axis=1)
                lab_len = len_lab_int

            labels = np.append(labels, [labels_int], axis=0)
            
    else:
        print("Error, labels_range must be type int or range")

    return labels

    
def load_images_fd(imgs_range, img_dir, normalise=True, img_type =".png", tensor=True, augment=True):
    
    
    if type(imgs_range)==int:
        img_file = img_dir + "/img_" + str(imgs_range) + img_type
        imgs = load_image_fd(img_file, normalise=normalise, img_type=img_type, augment=augment)
        imgs = tf.expand_dims(imgs, axis=0)
        
    elif type(imgs_range)==range:
        
        img_file = img_dir + "/img_" + str(imgs_range[0]) + img_type
        img = load_image_fd(img_file, normalise=normalise, img_type=img_type, augment=augment)
        imgs = tf.expand_dims(img, axis=0)
        
        for i in range(imgs_range[0]+1, imgs_range[-1]+1):
            img_file = img_dir + "/img_" + str(i) + img_type
            img = load_image_fd(img_file, normalise=normalise, img_type=img_type, augment=augment)
            img = tf.expand_dims(img, axis=0)
            imgs = tf.concat([imgs, img], axis=0)
            
    if tensor==False:
        sess = tf.Session()
        imgs = sess.run(imgs)
        sess.close()
    
    return imgs

def load_image_fd(img_file, normalise=True, img_type =".png", tensor=True, augment=True):
    """
    Loads the image required from the specified directory
    and normalises it
    
    """
    
    
    img = tf.read_file(img_file)
    
    if img_type==".png":
        img = tf.image.decode_png(img, channels=3)
    elif img_type==".jpg":
        img = tf.image.decode_jpeg(img, channels=3)
    else:
        print("Only png and jpg image types loadable")
        return

    
    if normalise==True:
        img = tf.divide(img, 255)
    
    
    if augment==True:
        img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        # Make sure the image is still in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
    
    
    if tensor==False:
        sess = tf.Session()
        img = sess.run(img)
        sess.close()

    return img


def assign_grid_box(labels, num_anchors = 3, num_scales = 3, st_gr_size = 13):
    """
    ARGS:
        labels: shape (batch_size, max num of labels, 4+num_classes)
    
    RETURNS:
        gr_coords: shape (batch_size, max num of labels, num_scales)
    
    Reads in labels and parameters
    Determines the starting reference value of each grid box at each scale
    within the detections tensor.
    These references and the references after depending on num_anchors
    should then be selected
    
    """

    batch_size = len(labels)
    lab_len = len(labels[0])
    gr_coords = np.zeros((batch_size,lab_len,num_scales*num_anchors), dtype=np.int32)

    for j in range(batch_size):
        
        labels_int = labels[j]
        scale_start = 0
        gr_size = st_gr_size
        
        for scale in range(num_scales):
            gr_len = 1/gr_size
            for i in range(lab_len):
                if sum(abs(labels_int[i]))!=0:
                    x = labels_int[i,1]
                    y = labels_int[i,2]
                    gr_x = int(x // gr_len)
                    gr_y = int(y // gr_len)
                    detect_ref = ((gr_y*gr_size) + gr_x)*num_anchors
                    detect_ref += scale_start
                    gr_coords[j,i,scale*num_anchors] = detect_ref
                    gr_coords[j,i,scale*num_anchors+1] = detect_ref+1
                    gr_coords[j,i,scale*num_anchors+2] = detect_ref+2
                    
            scale_start += (gr_size**2)*num_anchors
            gr_size = gr_size*2
            
    return gr_coords

#%% Post-processing Functions
    
def draw_boxes(box_params, img, img_file="detect_default.jpg"):
    """
    ARGS:
        box_params - (x_centre, y_centre, width, height)
        img - image to draw boxes on
        img_file - name of outputted image
    """
    
    half_w = box_params[:,2]/2
    half_h = box_params[:,3]/2
    y_tpl = box_params[:,1] - half_h
    x_tpl = box_params[:,0] - half_w
    y_btr = box_params[:,1] + half_h
    x_btr = box_params[:,0] + half_w
    
    boxes = np.stack((y_tpl, x_tpl, y_btr, x_btr), axis=1)
    boxes = np.expand_dims(boxes, axis=0)
    img = np.expand_dims(img, axis=0)
    
    if np.amax(img) <= 1:
        img = img*255
    
    detects_img = tf.image.draw_bounding_boxes(img, boxes)
    
    
    sess = tf.Session()
    detects_img = sess.run(detects_img)
    sess.close()
    
    detects_img = np.squeeze(detects_img, axis=0)
    cv2.imwrite(img_file, detects_img)

    return detects_img


def iou(box1, box2, mode='hw'):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    boxes -- list object with coordinates (x_tpl, y_tpl, x_btr, y_btr)
             or (x_centre, y_centre, width, height) if in hw mode
    
    """
    
    if mode=='hw': #convert coordinates to corners
        box1_n = [0,0,0,0]
        box2_n = [0,0,0,0]
        box1_n[0] = box1[0] - box1[2]
        box1_n[1] = box1[1] - box1[3]
        box2_n[0] = box2[0] - box2[2]
        box2_n[1] = box2[1] - box2[3]
        box1_n[2] = box1[0] + box1[2]
        box1_n[3] = box1[1] + box1[3]
        box2_n[2] = box2[0] + box2[2]
        box2_n[3] = box2[1] + box2[3]
        box1 = box1_n
        box2 = box2_n
    
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = max(yi2-yi1,0) * max(xi2-xi1,0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-10
    
    # compute the IoU
    iou = inter_area/union_area
    
    return iou



#%# Main
#%%
if __name__ == '__main__':
    out=train_yolo()
    print(out)