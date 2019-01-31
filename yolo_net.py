#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:44:01 2018
@author: Sean Wheeler


Creates the darknet-53 network using tensorflow
As found in 'YOLOv3: An Incremental Improvement by Joseph Redmon and Ali Farhadi'
In addition it builds the YOLO FPN-based detection layers on top of that.

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


#%% YOLO Network
def yolo_v3(inputs, num_classes, anchors, data_format='NCHW', is_training=False, batch_norm_epsilon=1e-05, batch_norm_decay=0.9,
            leaky_relu_alpha=0.1, reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = inputs / 255
    
    # set batch norm params
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    if is_training==True:
        biases_initializer=tf.zeros_initializer()
        weights_initializer=tf.contrib.layers.xavier_initializer()
    else:
        biases_initializer=None
        weights_initializer=None

        

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):
        
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params, 
                            biases_initializer=biases_initializer,
                            weights_initializer=weights_initializer,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=leaky_relu_alpha)):
            
            #Build Darknet-52 
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs, data_format)


            #Upsample final layer and concatenate with earlier layers for multi-scale detections 
            with tf.variable_scope('yolo-v3'):
                route, inputs = _yolo_block(inputs, 512, data_format=data_format)
                detect_1 = _detection_layer(inputs, num_classes, anchors[6:9], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')   #Lowest resolution detections (13^2 grid)
                
                inputs = _conv2d_fixed_padding(route, 256, 1, data_format=data_format)
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3) #Concatenate early darknet layer with upsampled final layer
                route, inputs = _yolo_block(inputs, 256, data_format=data_format)
                detect_2 = _detection_layer(inputs, num_classes, anchors[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')   #Middle resolution detections (26^2 grid)

                inputs = _conv2d_fixed_padding(route, 128, 1, data_format=data_format)
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3) #Concatenate early darknet layer with upsampled final layer
                _, inputs = _yolo_block(inputs, 128, data_format=data_format)
                detect_3 = _detection_layer(inputs, num_classes, anchors[0:3], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')   #Highest resolution detections (52^2 grid)


                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections





def _yolo_block(inputs, filters, data_format='NCHW'):
    """
    Builds a typical YOLO block in the detection layer, that effectively reduces
    the number of channels to the required input (2*filters)
    """
    
    inputs = _conv2d_fixed_padding(inputs, filters, 1, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, filters, 1, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, filters, 1, data_format=data_format)
    route = inputs
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3, data_format=data_format)
    
    return route, inputs




def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    
    #Create the final detection layer, where outputs = num of kernels for each grid cell
    detections = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,biases_initializer=tf.zeros_initializer())
    
    #Determine the size of the resolution (13, 26 or 52)
    grid_size = detections.get_shape().as_list()
    grid_size = grid_size[1:3] if (data_format=='NHWC') else grid_size[2:4]
    dim = grid_size[0] * grid_size[1]    #How many inputs to detection layer per channel
    box_attrs = 5 + num_classes          #How many outputs per box?
    
    if data_format == 'NCHW':
        detections = tf.reshape(detections, [-1, num_anchors*box_attrs, dim])
        detections = tf.linalg.transpose(detections)
    
    
    detections = tf.reshape(detections, [-1, num_anchors*dim, box_attrs])
    #Split the detections into the different categories
    #Centres(x,y), Sizes(w,h), Objectness, Class Logits (Softmaxed later)
    box_cens, box_sizs, box_objs, clss = tf.split(detections, [2, 2, 1, num_classes], axis=-1)
    
    #Create an array of reference points (one for each anchor per grid) 
    gr_x = tf.range(grid_size[0], dtype=tf.float32)
    gr_y = tf.range(grid_size[1], dtype=tf.float32)
    x_ref, y_ref = tf.meshgrid(gr_x, gr_y)
    x_ref = tf.reshape(x_ref, (-1,1))
    y_ref = tf.reshape(y_ref, (-1,1))
    gr_ref = tf.concat([x_ref, y_ref], axis=-1)
    gr_ref = tf.reshape( tf.tile(gr_ref, [1,num_anchors]) , [1, -1, 2])
    
    #Side lengths of a grid box in pixels of input image 
    grid_len = (img_size[0] // grid_size[0], img_size[1] // grid_size[1]) 
    #Normalise the anchor lengths by the grid box sides
    anchors = [(anchor[0] / grid_len[0], anchor[1] / grid_len[1]) for anchor in anchors]

    #Process the network outputs sigma(t_x) +c_x
    box_cens = tf.multiply( tf.add( tf.nn.sigmoid(box_cens) , gr_ref), grid_len)
    
    anchors = tf.tile(anchors, [dim,1])
    box_sizs = tf.multiply( tf.multiply( tf.exp(box_sizs), anchors), grid_len)                   
                      
    box_objs = tf.nn.sigmoid(box_objs)
    
    detections = tf.concat([box_cens, box_sizs, box_objs, clss], axis=-1)
    
    return detections




def _upsample(inputs, out_shape, data_format='NHWC'):
    
    if data_format =='NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        height_n = out_shape[3]
        width_n = out_shape[2]
    else:
        height_n = out_shape[2]
        width_n = out_shape[1]
        
    inputs = tf.image.resize_nearest_neighbor(inputs, (height_n, width_n))
    
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        
    inputs = tf.identity(inputs, name='upsampled')
    
    return inputs

#%% Darknet

@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, data_format = 'NCHW', mode='CONSTANT', **kwargs):
    
    """Pads input H and W with a fixed amount of padding, independent of input size.
    
    Arguments::
        inputs - tensor, [batch, C, H, W] or [batch, H, W, C]
        kernel_size - positive integer, kernel to be used in conv2d or max_pool2d
        mode - sring, the mode for tf.pad
        
    Returns::
        padded_inputs - tensor, same format as inputs and padded if kernel_size > 1
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    if data_format == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0,0], [0,0], [pad_beg,pad_end], [pad_beg,pad_end]], mode = mode)
    else:
        padded_inputs = tf.pad(inputs,  [[0,0], [pad_beg,pad_end], [pad_beg,pad_end], [0,0]], mode = mode)
    return padded_inputs



def _conv2d_fixed_padding(inputs, filters, kernel_size, strides = 1, data_format='NCHW'):
    """
    Adds a 2D convolutional layer to inputs and pads the image if necessary
    """
    
    if (strides > 1):     #If layer needs fixed padding
        inputs = _fixed_padding(inputs, kernel_size, data_format = data_format)
        
    inputs = slim.conv2d(inputs, filters, kernel_size, stride = strides, 
                              padding =( 'SAME' if strides == 1 else 'VALID'))
    
    return inputs


def _darknet_53_block(inputs, filters, num_blocks=1, data_format='NCHW'):
    """
    Constructs typical blocks used in Darknet consisting of two conv layers
    and a residual throughput
    """
    for i in range(num_blocks):
        residual = inputs
        inputs = _conv2d_fixed_padding(inputs, filters, 1, data_format=data_format)
        inputs = _conv2d_fixed_padding(inputs, 2*filters, 3, data_format=data_format)
        inputs = inputs + residual
    
    return inputs


def darknet53(inputs, data_format='NCHW'):
    """
    Builds the darknet model not including the avgpool, connected or softmax layers
    also returns the outputs at 2 additional scales
    for the FPN-inspired detection module
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2, data_format=data_format)
    inputs = _darknet_53_block(inputs, 32, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2, data_format=data_format)
    inputs = _darknet_53_block(inputs, 64, num_blocks=2, data_format=data_format)
    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2, data_format=data_format)
    inputs = _darknet_53_block(inputs, 128, num_blocks=8, data_format=data_format)
    scale_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2, data_format=data_format)
    inputs = _darknet_53_block(inputs, 256, num_blocks=8, data_format=data_format)
    scale_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2, data_format=data_format)
    inputs = _darknet_53_block(inputs, 512, num_blocks=4, data_format=data_format)
    
    return scale_1, scale_2, inputs


#%% TF Assign Label function and TF Loops
def tf_assign_label(labels, labels_grids, detections, iou_threshold=0.5, num_scs=3, num_ancs=3, num_classes=9):
    
    """
    ARGS:
        labels = (x, y, h, w, classes) - tensor shape (num_batches, num_labels, 4+num_classes)
        labels_grids = grid box indices - tensor shape (num_batches, num_labels, num_scales*num_anchors)
        detections = outputted detections from YOLOv3. 
                        Shape: (num_batches, num_predicted_boxes, 4+1+num_classes)
        where  for num_scales, num_anchors = 3: 10647 = (13^2 + 26^2 + 52^2)*3
        
    RETURNS:
        labels_assigned = indices of bounding box detections assigned to labels
                        Elements: (batch_num, label_num, assigned_pred_ind)
                        Shape: (num_assigned_labels, 3)     
        obj_present = indices of bounding box detections assigned/or with iou
                        over threshold. Elements: (batch_num, assigned_pred_ind)
                        Shape: (num_assigned_labels+num_labels_over_threshold, 2)
        
        
    This function calculates IoUs between the detections and labels within
    the relevant grid box. It then assigns the detections to the labels with
    the highest IoU. If there is any detection with no IoU with any label it is
    then assigned to the label which it's centre is nearest to.
    
    """
    
    #Determine parameters for loop iterations        
    pred_size = detections.get_shape()
    batch_size = labels.get_shape()[0]
    labels_size = tf.shape(labels)[1]
    num_pos_ancs = num_ancs*num_scs
    
    #Execute IoU loop which calculates the IoUs with possible anchors for each label and stores in tensor of shape
    #(batch_size, labels_size, num_pos_anchors, 2) where the final dimension contains (iou, bounding_box_index)
    batch_num=0
    loop_vars = [batch_num, batch_size, labels_size, num_pos_ancs, detections,
                 labels_grids, labels, tf.zeros([batch_size, labels_size, num_pos_ancs, 2])]
    
    con_shp = tf.constant(0).get_shape()
    shp_invars = [con_shp, con_shp, con_shp, con_shp, pred_size, 
                  labels_grids.get_shape(), labels.get_shape(), tf.TensorShape([None,None,None, None])]
   
    batch_iou_out=tf.while_loop(_tf_count, _batch_iou_loop, loop_vars, shp_invars, back_prop=False)
    
    pos_ious = batch_iou_out[-1]
    ious, refs = tf.split(pos_ious, 2, axis=3)
    ious = tf.layers.flatten(ious)
    refs = tf.layers.flatten(refs)  

    #Execute loop which assigns each label an anchor, provided it has an IoU>0
    batch_num=0
    loop_vars = [batch_num, batch_size, labels_size, num_pos_ancs, 
                 ious, refs, tf.zeros([batch_size, labels_size])]
                 
    shp_invars = [con_shp, con_shp, con_shp, con_shp, 
                  tf.TensorShape([None,None]), tf.TensorShape([None,None]), 
                  tf.TensorShape([None,None])]
   
    batch_assign_out=tf.while_loop(_tf_count, _batch_assign_loop, loop_vars, shp_invars, back_prop=False)
    labels_assigned = tf.cast(batch_assign_out[-1], tf.int32)
    
    #Any bounding box with an iou less than the threshold is marked with -1
    obj_present = tf.cast(tf.greater_equal(ious, iou_threshold), tf.float32)
    obj_present_no = obj_present-1
    obj_present = tf.cast(refs*obj_present+obj_present_no, tf.int32)

    #Create tensors which concatenates batch numbers on labels_assigned and obj_present
    rang = tf.range(batch_size)
    rang = tf.reshape(rang, [-1,1])
    rang_1 = tf.tile(rang, [1,labels_size])
    rang_2 = tf.tile(rang, [1,tf.shape(obj_present)[1]])

    #Create masks to be remove any -1 elements
    mask_1 = tf.not_equal(labels_assigned, -1)
    mask_2 = tf.not_equal(obj_present, -1)
    
    #Expand dimensions for concatenation
    labels_assigned = tf.expand_dims(labels_assigned, axis=2)
    obj_present = tf.expand_dims(obj_present, axis=2)
    rang_1 = tf.expand_dims(rang_1, axis=2)
    rang_2 = tf.expand_dims(rang_2, axis=2)
    
    #Create tensor which concatenates label numbers on labels_assigned
    label_nums = tf.range(labels_size)
    label_nums = tf.tile(label_nums, [batch_size])
    label_nums = tf.reshape(label_nums, [batch_size, labels_size])
    label_nums = tf.expand_dims(label_nums, axis=2)

    #Add label and batch numbers to label_assigned and batch numbers to obj_present
    labels_assigned = tf.concat([label_nums, rang_1, labels_assigned],axis=2)
    obj_present = tf.concat([rang_2, obj_present],axis=2)
    
    #Apply the boolean masks to remove -1 elements which represent 
    #labels with no assigned bounding box for labels_assigned
    #bounding boxes with IoU less than threshold for obj_present
    labels_assigned = tf.boolean_mask(labels_assigned,mask_1,axis=0)
    obj_present = tf.boolean_mask(obj_present,mask_2,axis=0)

    #Add all assigned bounding boxes to thresholded bounding boxes
    obj_present = tf.concat([obj_present, labels_assigned[:,1:3]],axis=0)
    obj_present = _tf_unique_2d(obj_present) #Remove multiple appearances of the same bounding box

    return labels_assigned, obj_present


def _batch_iou_loop(batch_num, batch_size, labels_size, num_pos_ancs, detections, labels_grids, labels, pos_ious):
    """
    ARGS:
        - batch_num & batch_size = counter and limit for loop
        - labels_size = limit for nested label loop
        - num_pos_ancs = number of possible anchors for each label
        - labels = (x, y, h, w, classes) - tensor shape (num_batches, num_labels, 4+num_classes)
        - labels_grids = grid box indices - tensor shape (num_batches, num_labels, num_scales*num_anchors)
        - detections = outputted detections from YOLOv3. 
                        Shape: (num_batches, num_predicted_boxes, 4+1+num_classes)
    RETURNS:
        - pos_ious = tensor containing IoUs of labels with possible bounding box detections
                    Elements: (IoU, bounding box detection index)
                    Shape: (batch_size, label_size, num_pos_ancs, 2)

    """
    #Loop over each image in the batch
    con_shp = tf.constant(0).get_shape()
    img_labels_num=0
    loop_vars = [img_labels_num, labels_size, num_pos_ancs, detections[batch_num],
                 labels_grids[batch_num], labels[batch_num],
                 tf.zeros([labels_size, num_pos_ancs, 14])]

    shp_invars = [con_shp, con_shp, con_shp, detections[batch_num].get_shape(),
                  labels_grids[batch_num].get_shape(), labels[batch_num].get_shape(),
                  tf.TensorShape([None,None,None])]
    
    labels_loop_out = tf.while_loop(_tf_count, _labels_iou_loop, 
                                 loop_vars, shp_invars, back_prop=False)
    
    img_pos_ious = labels_loop_out[-1]
    img_pos_ious = tf.expand_dims(img_pos_ious, axis=0)
    
    #Add all possible anchors to tensor
    pos_ious = tf.cond( tf.equal(batch_num,0), 
                       lambda: img_pos_ious,
                       lambda: tf.concat([pos_ious, img_pos_ious],0) )
    
    return batch_num+1, batch_size, labels_size, num_pos_ancs, detections, labels_grids, labels, pos_ious


def _labels_iou_loop(img_labels_num, labels_size, num_pos_ancs, img_detections, img_labels_grids, img_labels, img_pos_ious):
    """
    Nested loop inside batch_iou_loop
    ARGS:
        - img_labels_num & labels_size = counter and limit for loop
        - num_pos_ancs = number of possible anchors for each label
        - img_labels = (x, y, h, w, classes) - tensor shape (num_labels, 4+num_classes)
        - img_labels_grids = grid box indices - tensor shape (num_labels, num_scales*num_anchors)
        - img_detections = outputted detections from YOLOv3. 
                            Shape: (num_predicted_boxes, 4+1+num_classes)
    RETURNS:
        - img_pos_ious = tensor containing IoUs of labels with possible bounding box detections
                    Elements: (IoU, bounding box detection index)
                    Shape: (label_size, num_pos_ancs, 2)

    """
    #Loop over each label for an image
    con_shp = tf.constant(0).get_shape()
    label_num = 0
    loop_vars = [label_num, num_pos_ancs, img_detections,
                 img_labels_grids[img_labels_num], img_labels[img_labels_num],
                 tf.zeros([num_pos_ancs, 14])]
    
    shp_invars = [con_shp, con_shp, img_detections.get_shape(), 
                  img_labels_grids[img_labels_num].get_shape(), img_labels[img_labels_num].get_shape(),
                  tf.TensorShape([None, None])]
    
    label_loop_out = tf.while_loop(_tf_count, _label_iou_loop,
                                   loop_vars, shp_invars, back_prop=False)
      
    #Extract the possible anchors for each ground truth
    lab_ious = label_loop_out[-1]
    lab_ious = tf.expand_dims(lab_ious, axis=0)
    
    #Add all possible anchors to tensor
    img_pos_ious = tf.cond( tf.equal(img_labels_num,0),
                                lambda: lab_ious, 
                                lambda: tf.concat([img_pos_ious, lab_ious],0) )
    
    return img_labels_num+1, labels_size, num_pos_ancs, img_detections, img_labels_grids, img_labels, img_pos_ious


def _label_iou_loop(label_num, num_pos_ancs, img_detections, label_grids, label, lab_ious):
    """
    Nested loop inside labels_iou_loop
    ARGS:
        - label_num & num_pos_ancs = counter and limit for loop
        - label = (x, y, h, w, classes) - tensor shape (4+num_classes)
        - label_grids = grid box indices - tensor shape (num_scales*num_anchors)
        - img_detections = outputted detections from YOLOv3. 
                            Shape: (num_predicted_boxes, 4+1+num_classes)
    RETURNS:
        - lab_ious = tensor containing IoUs of labels with possible bounding box detections
                    Elements: (IoU, bounding box detection index)
                    Shape: (num_pos_ancs, 2)

    """
    #Gather all the appropriate anchor detections
    lab_pos_ancs = tf.gather(img_detections, label_grids)
    lab_pos_ancs = lab_pos_ancs[:,0:4]
    lab_box_coords = label[0:4]
    lab_boxes = tf.tile(lab_box_coords, [tf.constant(9)])
    lab_boxes = tf.reshape(lab_boxes, (num_pos_ancs,4))
    
    #Calculate the IoUs and concatenate with bounding box detection indices
    lab_ious = tf_iou(lab_boxes, lab_pos_ancs)
    lab_gr_inds = tf.expand_dims(tf.to_float(tf.transpose(label_grids)),axis=1)
    #Concatenate possible anchors with their index in detections to use later
    lab_ious = tf.concat([lab_ious, lab_gr_inds], axis=1)

    return label_num+1, num_pos_ancs, img_detections, label_grids, label, lab_ious


def _batch_assign_loop(batch_num, batch_size, labels_size, num_pos_ancs, ious, refs, labels_assigned):
    """
    ARGS:
        - batch_num & batch_size = counter and limit for loop
        - labels_size = limit for nested label loop
        - num_pos_ancs = number of possible anchors for each label
        - ious = tensor containing IoUs of labels with possible bounding box detections
                    Elements: (IoU)
                    Shape: (batch_size, label_size, num_pos_ancs)
        - refs = tensor containing indexes of bounding box detections which were used
                 to calculate the IoUs in the iou tensor.
                    Elements: (bounding box detection index)
                    Shape: (batch_size, label_size, num_pos_ancs)
    RETURNS:
        - labels_assigned = assigns each label it can a unique bounding box based
                            on the highest IoUs
                    Elements: (assigned bounding box detection index)
                    Shape: (batch_size, label_size)

    """
    #Labels Loop    
    assign_comp=False

    loop_vars = [assign_comp, labels_size, num_pos_ancs, 
                 ious[batch_num], refs[batch_num],
                  batch_num, tf.zeros(labels_size)-1]
        
    labels_assign_loop_out = tf.while_loop(_tf_bool, _labels_assign_loop, 
                                 loop_vars, back_prop=False)
    
    img_labels_assigned = labels_assign_loop_out[-1]
    img_labels_assigned = tf.expand_dims(img_labels_assigned, axis=0)
    
    #Add all possible anchors to tensor
    labels_assigned = tf.cond( tf.equal(batch_num,0), 
                       lambda: img_labels_assigned,
                       lambda: tf.concat([labels_assigned, img_labels_assigned],0) )

    return batch_num+1, batch_size, labels_size, num_pos_ancs, ious, refs, labels_assigned


def _labels_assign_loop(assign_comp, labels_size, num_pos_ancs, ious, refs, batch_num, img_labels_assigned):
    """
    ARGS:
        - assign_comp = condition for loop
        - labels_size = limit for nested label loop
        - num_pos_ancs = number of possible anchors for each label
        - ious = tensor containing IoUs of labels with possible bounding box detections
                    Elements: (IoU)
                    Shape: (batch_size, label_size, num_pos_ancs)
        - refs = tensor containing indexes of bounding box detections which were used
                 to calculate the IoUs in the iou tensor.
                    Elements: (bounding box detection index)
                    Shape: (batch_size, label_size, num_pos_ancs)
    RETURNS:
        - labels_assigned = assigns each label it can a unique bounding box based
                            on the highest IoUs
                    Elements: (assigned bounding box detection index)
                    Shape: (label_size)       

    """

    tot_pos_ancs = labels_size*num_pos_ancs
    
    #Get max IoU value
    max_iou_ref = tf.argmax(ious, axis=0)
    max_iou = ious[max_iou_ref]
    max_iou_box = refs[max_iou_ref]


    #if max iou = 0 then assignation for image complete and break from loop
    assign_comp = tf.cond( tf.less_equal(max_iou,0),
                   lambda: True,
                   lambda: False )
    
    #check if bounding box already assigned
    max_iou_box = tf.tile([max_iou_box], [labels_size])
    assigned = tf.equal(max_iou_box, img_labels_assigned)
    assigned = tf.reduce_sum(tf.cast(assigned, tf.float32))
    assigned = tf.equal(assigned,1.0)
    
    #If bounding box assigned then zero that iou
    ious = tf.cond( assigned, 
                     lambda: ious*_tf_zero_mask(max_iou_ref, tot_pos_ancs),
                     lambda: ious)
    

    
    #if box unassigned and IoU>0 assign it to that label in img_labels_assigned and zero all ious for that label
    assign_label_cond = tf.logical_and( tf.equal(assign_comp, False), tf.equal(assigned, False))
    img_labels_assigned, ious = tf.cond(assign_label_cond, 
                                  lambda: _label_assign_function(max_iou_ref, max_iou_box,
                                                                num_pos_ancs, labels_size,
                                                                tot_pos_ancs, img_labels_assigned,
                                                                ious),
                                  lambda: (img_labels_assigned, ious))

    return assign_comp, labels_size, num_pos_ancs, ious, refs, batch_num, img_labels_assigned


def _label_assign_function(max_iou_ref, max_iou_box, num_pos_ancs, labels_size, tot_pos_ancs, img_labels_assigned, ious ):
    
    num_pos_ancs = tf.cast(num_pos_ancs, tf.int64)
    labels_size = tf.cast(labels_size, tf.int64)
    
    #Determine which label the bounding box is associated with
    label_num = max_iou_ref//num_pos_ancs
    #Store the assigned bounding box index in img_labels_assigned
    img_labels_assigned = img_labels_assigned + _tf_one_mask(label_num, labels_size, max_iou_box+1)
    #As label is now assigned, zero all it's IoUs in the iou tensor
    zero_mask = _tf_zero_mask(label_num*num_pos_ancs, tot_pos_ancs, num_pos_ancs)
    ious = ious * zero_mask
    
    return img_labels_assigned, ious

#%%General TF functions

def tf_iou(box1, box2, mode='hw'):
    """
    ARGS: 
        box1, box2 = tensors containing box parameters and depending on mode: 
                    Elements: (x,y,w,h) or (x_tpl, y_tpl, x_btr, y_btr)
                    Shape: (num_boxes, 4)
    RETURNS: 
        tf_ious = tensor containing the IoUs of the inputs
                  Shape: (num_boxes)
    
    """

    box1 = tf.to_float(box1)
    box2 = tf.to_float(box2)
        
    if mode=='hw':
        xc_1, yc_1, w_1, h_1 = tf.split(box1, 4, axis=1)
        xc_2, yc_2, w_2, h_2 = tf.split(box2, 4, axis=1)
    
        xtpl_1 = xc_1 - w_1
        ytpl_1 = yc_1 - h_1
        xbtr_1 = xc_1 + w_1
        ybtr_1 = yc_1 + h_1
        
        xtpl_2 = xc_2 - w_2
        ytpl_2 = yc_2 - h_2
        xbtr_2 = xc_2 + w_2
        ybtr_2 = yc_2 + h_2
    else:
        xtpl_1, ytpl_1, xbtr_1, ybtr_1 = tf.split(box1, 4, axis=1)
        xtpl_2, ytpl_2, xbtr_2, ybtr_2 = tf.split(box2, 4, axis=1)
    
    
    xi1 = tf.maximum(xtpl_1,xtpl_2)
    yi1 = tf.maximum(ytpl_1,ytpl_2)
    xi2 = tf.minimum(xbtr_1,xbtr_2)
    yi2 = tf.minimum(ybtr_1,ybtr_2)
    inter_area = tf.maximum(yi2-yi1,0) * tf.maximum(xi2-xi1,0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (xbtr_1-xtpl_1) * (ybtr_1-ytpl_1)
    box2_area = (xbtr_2-xtpl_2) * (ybtr_2-ytpl_2)
    union_area = box1_area + box2_area - inter_area + 1e-10
    
    # compute the IoU
    tf_ious = inter_area/union_area
    
    return tf_ious

def _tf_unique_2d(x):
    """
    ARGS: 
        X = shape: 2d tensor potentially with elements with the same value
    RETURNS: 
        X = shape: 2d tensor with all values being unique
    
    """
    
    x_shape=tf.shape(x)
    x1=tf.tile(x,[1,x_shape[0]]) 
    x2=tf.tile(x,[x_shape[0],1]) 

    x1_2 = tf.reshape(x1,[x_shape[0]*x_shape[0],x_shape[1]])
    x2_2 = tf.reshape(x2,[x_shape[0]*x_shape[0],x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2,x2_2),axis=1)
    cond = tf.reshape(cond,[x_shape[0],x_shape[0]]) 
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond,tf.int32) 
    cond_zeros = tf.zeros(cond_shape,tf.int32) 

    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r,[x_shape[0]]),1)
    r = tf.reshape(r,[x_shape[0],x_shape[0]])

    f1 = tf.multiply(tf.ones(cond_shape,tf.int32),x_shape[0]+1)
    f2 = tf.ones(cond_shape,tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast,cond_zeros),f1,f2)

    r_cond_mul = tf.multiply(r,cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul,axis=1)
    r_cond_mul3,unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3,1)

    x=tf.gather(x,r_cond_mul4)

    return x

def _tf_zero_mask(zero_st, tensor_len, length=1):
    #Creates a tensor of ones with elements of zeros with a given length at a specified position
    tensor_len = tf.cast(tensor_len, tf.int64)
    top = tf.ones(zero_st)
    mid = tf.zeros(length)
    bot = tf.ones(tensor_len-zero_st-length)
    mask = tf.concat([top,mid,bot],axis=0)
    return mask

def _tf_one_mask(one_st, tensor_len, value=1, length=1):
    #Creates a tensor of zeros with elements of a given value with a given length at a specified position
    tensor_len = tf.cast(tensor_len, tf.int64)
    top = tf.zeros(one_st)
    mid = tf.ones(length)
    bot = tf.zeros(tensor_len-one_st-length)
    mask = tf.concat([top,mid,bot],axis=0)
    mask = mask*value
    return mask

def _tf_bool(i, *args):
    return tf.equal(i, False)

def _tf_count(i, max_count=10, *args):
    return tf.less(i, max_count)
