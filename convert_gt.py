# -*- coding: utf-8 -*-

"""
Copyright 15 November 2018, Sean Wheeler, All rights reserved

This file reads in the groundtruth data from the ICDAR2017 MLT dataset and converts it
into the desired format. It resizes the ground truths to rectangles and determines
the height and width of these boxes

In addition it is capable of pre-processing the images to the desired size and will adjust
the ground truth accordingly
"""


import csv
import cv2
import os
import tensorflow as tf


imgs_len = 7200
dataset = 'training'
img_h_new = 416
img_w_new = 416

input_image_dir = dataset + '_data/'
output_image_dir = "processed_data/" + dataset + "_images/"

input_label_dir = dataset + '_localization_data/gt_img_'
output_label_dir = "processed_data/" + dataset + "_labels/"

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

scripts = {"Arabic":0,
           "Latin":1,
           "Chinese":2,
           "Japanese":3,
           "Korean":4,
           "Bangla":5,
           "Symbols":6,
           "Mixed":7,
           "None":8
           }




for img_num in range(1,imgs_len + 1):
    
    """
    Resizes images to 416x416 for YOLO training
    Reads in individual ground truths
    Recalculates for downsampled images
    Changes all groundtruths into rectangles for usage in YOLO
    Can calculate area of each groundtruth"""
    
    gt_output = []

    img = cv2.imread(input_image_dir + "img_" + str(img_num) + '.jpg')
    img_h, img_w, _ = img.shape
    img_resized = cv2.resize(img, (img_h_new, img_w_new), interpolation=cv2.INTER_CUBIC)
    output_image_file = output_image_dir + "img_" + str(img_num) + '.png'
    cv2.imwrite(output_image_file, img_resized)
    ratio_h = 1/img_h  #used to normalise labels to range [0:1]
    ratio_w = 1/img_w

    input_label_file = input_label_dir + str(img_num) + '.txt'
    with open(input_label_file, newline='') as input_file:
        for row in csv.reader(input_file):
            
            for i in range(0,8):
                row[i]=float(row[i])   

            x_tpl = min([row[0],row[2],row[4],row[6]])*ratio_w
            y_tpl = min([row[1],row[3],row[5],row[7]])*ratio_h
            x_btr = max([row[0],row[2],row[4],row[6]])*ratio_w
            y_btr = max([row[1],row[3],row[5],row[7]])*ratio_h
            width = x_btr - x_tpl
            height = y_btr - y_tpl
            x_centre = x_tpl + width/2
            y_centre = y_tpl + height/2
            one_hot = np.zeros(9, dtype=np.uint8)
            one_hot[scripts[row[8]]]=1
            #area  = x_height * y_height    #For determining anchor boxes with K-means
            gt_output.append([x_centre, y_centre, width, height,one_hot])
    
    #Writes out file containing all readjusted groundtruths
    output_label_file = output_label_dir + "label_" + str(img_num) + '.txt'
    with open(output_label_file, "w") as output_file:
        writer = csv.writer(output_file)    
        writer.writerows(gt_output)
        

