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

#Create three empty arrays for the data
boxes = []


with open("localization_sorted.csv") as f:
    for row in csv.reader(f):
        
        for i in range(3,7):
            row[i]=int(row[i])
        
        box_width = row[5]-row[3]
        box_height = row[6]-row[4]
        boxes.append([box_width, box_height])
            

boxes = np.array(boxes, dtype=float)     

anchor_boxes, distortion_1 = sp.vq.kmeans(boxes, 9, iter=300)

np.savetxt("anchor_boxes.txt", anchor_boxes.astype(int), fmt='%i', delimiter=",")
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
    
    
sort_csv("training_localization_data_resized.txt", localization_data_types, 7)

#%%