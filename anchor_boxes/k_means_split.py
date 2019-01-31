#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:44:09 2018

@author: aegeus
"""

import csv
import operator
import scipy.cluster as sp
import numpy as np

localization_data_types = (int, int, int, int, int, int, int, int, str, str)

word_total = 86183
split_point_1 = round(86183/3)
split_point_2 = 2 * split_point_1

#Create three empty arrays for the data
split_1 = []
split_2 = []
split_3 = []

count = 0
with open("localization_sorted.csv") as f:
    for row in csv.reader(f):
        
        
        for i in range(3,7):
            row[i]=int(row[i])
        
        box_width = row[5]-row[3]
        box_height = row[6]-row[4]
        
        if count < split_point_1:
            split_1.append([box_width, box_height])
        elif count < split_point_2:
            split_2.append([box_width, box_height])
        else:
            split_3.append([box_width, box_height])
            
        count += 1

split_1 = np.array(split_1, dtype=float)     
split_2 = np.array(split_2, dtype=float)     
split_3 = np.array(split_3, dtype=float)     

anchor_boxes_1, distortion_1 = sp.vq.kmeans(split_1, 3, iter=300)
anchor_boxes_2, distortion_2 = sp.vq.kmeans(split_2, 3, iter=300)
anchor_boxes_3, distortion_3 = sp.vq.kmeans(split_3, 4, iter=300)


anchor_boxes = np.concatenate((np.round(anchor_boxes_1),np.round(anchor_boxes_2),np.round(anchor_boxes_3)))

np.savetxt("anchor_boxes_split.txt", anchor_boxes.astype(int), fmt='%i', delimiter=",")
#%%

def sort_csv(csv_filename, types, sort_columns):
    data = []
    
    with open(csv_filename) as f:
        for row in csv.reader(f):
            data.append(convert(types,row))
    
    data.sort(key=operator.itemgetter(sort_columns))
    
    with open ("localization_sorted.csv", 'w') as f:
        csv.writer(f).writerows(data)
        
        
def convert(convert_funcs, seq):
    return [item if func is None else func(item)
            for func, item in zip(convert_funcs,seq)]
    
    
#sort_csv("training_localization_data_resized.txt", localization_data_types, 7)

#%%