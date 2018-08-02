# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:35:42 2018

@author: Leonardo
"""

import numpy as np

def Detectionv4(class_detected,box_detected,true_det,counter,counter2,location_x,limit,true_det_past):
    aux=range(len(class_detected))
    area_light=0.03
    area_vert_signal=0.08
    area_pedestrian=0.6
    area_cebra=0.8
    deviation=0.25
    counter_past=np.array(counter)
     #Counter to add elements
    for i in aux:
       [ymin,xmin,ymax,xmax]=box_detected[i]
       area=(ymax-ymin)*(xmax-xmin)*100
       if class_detected[i]=='rojo':
           if location_x[0]==0:
               location_x[0]=xmin 
           if area >= area_light and (xmin<deviation+location_x[0] and xmin>location_x[0]-deviation) and true_det_past[0]!='rojo' and true_det_past[0]!='verde' and true_det_past[0]!='amarillo':
               counter[0]=counter[0]+1
               if counter[0]>=limit and counter[1]<limit and counter[2]<limit: #priority to other SEMOFORO color
                   true_det=[class_detected[i]]
       elif class_detected[i]=='amarillo':
           if location_x[1]==0:
               location_x[1]=xmin 
           if area >= area_light and (xmin<deviation+location_x[1] and xmin>location_x[1]-deviation):
               counter[1]=counter[1]+1
               if counter[1]==limit : #without restrictions
                   true_det=[class_detected[i]]
       elif class_detected[i]=='verde':
           if location_x[2]==0:
               location_x[2]=xmin 
           if area >= area_light and (xmin<deviation+location_x[2] and xmin>location_x[2]-deviation) and true_det_past[0]!='verde' and true_det_past[0]!='amarillo':
               counter[2]=counter[2]+1
               if counter[2]==limit and counter[1]<limit and counter[0]<limit : #priority to other SEMOFORO color
                   true_det=[class_detected[i]]
       elif class_detected[i]=='pare':
           if location_x[3]==0:
               location_x[3]=xmin    
           if area >= area_vert_signal and (xmin<deviation+location_x[3] and xmin>location_x[3]-deviation): 
               counter[3]=counter[3]+1
               if counter[3]==limit and (counter[0]<limit and counter[1]<limit and counter[2]<limit): #priority to SEMAFORO
                   true_det=[class_detected[i]]
       elif class_detected[i]=='cruce':
           if location_x[4]==0:
               location_x[4]=xmin 
           if area >= area_vert_signal and (xmin<deviation+location_x[4] and xmin>location_x[4]-deviation):
               counter[4]=counter[4]+1
               if counter[4]==limit and (counter[0]<limit and counter[1]<limit and counter[2]<limit and counter[3]<limit): #priority to SEMAFORO
                   true_det=[class_detected[i]]
       elif class_detected[i]=='cebra':
           if location_x[5]==0:
               location_x[5]=xmin 
           if area >= area_cebra and ymax<=0.9 and (xmin<deviation+location_x[5] and xmin>location_x[5]-deviation): 
               counter[5]=counter[5]+1
               if counter[5]==limit and (counter[0]<limit and counter[1]<limit and counter[2]<limit and counter[3]<limit): #priority to SEMAFORO
                   true_det=[class_detected[i]]
       elif class_detected[i]=='ceda':
           if location_x[6]==0:
               location_x[6]=xmin 
           if area >= area_vert_signal and (xmin<deviation+location_x[6] and xmin>location_x[6]-deviation):
               counter[6]=counter[6]+1
               if counter[6]==limit and (counter[0]<limit and counter[1]<limit and counter[2]<limit and counter[3]<limit): #priority to SEMAFORO
                   true_det=[class_detected[i]]
       elif class_detected[i]=='peaton':
           if location_x[7]==0:
               location_x[7]=xmin 
           if area >= area_pedestrian and (xmin<deviation+location_x[7] and xmin>location_x[7]-deviation):
               counter[7]=counter[7]+1
               if counter[7]>=limit and true_det!=[] and len(true_det)<=1: #only added one time if another signal was detected
                   true_det=true_det+[class_detected[i]]
    
    #Loop to wax elements when a long time has passed without detections
    for k in range(len(counter)):
        if counter[k]==counter_past[k]:
            counter2[k]=counter2[k]+1
            if counter2[k]==limit*limit:
                counter[k]=0
                counter2[k]=0
                location_x[k]=0
        else:
            counter2[k]=0
    return true_det,counter,counter2,location_x

