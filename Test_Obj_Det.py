# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:39:15 2018

@author: Leonardo
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools

def Test_Obj_Det(class_detected_buffer,box_detected_buffer):
    #Extract_Ground Truth
    with open('images/' +'confusion_labels.csv') as File:
    #with open('images/' +'test_labels.csv') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        count=0
        image_name=[]
        obj_truth=[]
        x_truth=[]
        y_truth=[]
        for row in reader:
            if count!=0:
                width=float(row[1])
                height=float(row[2])
                image_name=image_name+[row[0]]
                obj_truth=obj_truth+[(row[3])]
                x_truth=x_truth+[((float(row[4])+float(row[6]))/2)/width]
                y_truth=y_truth+[((float(row[5])+float(row[7]))/2)/height]
            count=1
        name_past=image_name[0]
        count=0 
        aux=[]
        aux2=[]
        aux3=[]
        class_truth_buffer=[]
        x_truth_buffer=[]
        y_truth_buffer=[]
        for name in image_name:
            if name==name_past:
                aux=aux+[obj_truth[count]]
                aux2=aux2+[x_truth[count]]
                aux3=aux3+[y_truth[count]]
            else:
                class_truth_buffer=class_truth_buffer+[aux]
                x_truth_buffer=x_truth_buffer+[aux2]
                y_truth_buffer=y_truth_buffer+[aux3]
                name_past=name
                aux=[obj_truth[count]]
                aux2=[x_truth[count]]
                aux3=[y_truth[count]]
            count=count+1
        class_truth_buffer=class_truth_buffer+[aux]
        x_truth_buffer=x_truth_buffer+[aux2]
        y_truth_buffer=y_truth_buffer+[aux3]
        
               
        #Confusion Matrix
        deviation=0.1
        confusion_m=np.zeros((9,8))
        confusion_ref=np.zeros((8,8))
        confusion_norm=np.zeros((9,8))
        #loop with number of frames
        for frame in range(len(class_truth_buffer)):
            class_detected=class_detected_buffer[frame]
            box_detected=box_detected_buffer[frame]
            class_truth=class_truth_buffer[frame]
            box_truth_x=x_truth_buffer[frame]
            box_truth_y=y_truth_buffer[frame]
            class_ref=class_truth
            box_ref_x=box_truth_x
            box_ref_y=box_truth_y
            #Loop of truth objects labelled
            for i in range(len(class_truth)):
                x_truth=box_truth_x[i]
                y_truth=box_truth_y[i]
                if class_truth[i]=='rojo':
                    column=0
                elif class_truth[i]=='amarillo':
                    column=1
                elif class_truth[i]=='verde':
                    column=2
                elif class_truth[i]=='ceda':
                    column=3
                elif class_truth[i]=='cruce':
                    column=4
                elif class_truth[i]=='cebra':
                    column=5
                elif class_truth[i]=='pare':
                    column=6
                elif class_truth[i]=='peaton':
                    column=7
                #elif class_truth[i]=='vehiculo':
                    #column=8
               # print('Columna:%d'%(column))
                #Loop of detected objects
                for j in range(len(class_detected)):
                    [ymin,xmin,ymax,xmax]=box_detected[j]
                    x_detected=(xmin+xmax)/2
                    y_detected=(ymin+ymax)/2
                    if class_detected[j]=='cebra' and column==5:
                        print('x_detected:%f y_detected:%f'%(x_detected,y_detected))
                        devia_aux=0.05
                    else:
                        devia_aux=0
                    if (x_detected<=x_truth+deviation+devia_aux) and (x_detected>=x_truth-deviation-devia_aux) and (y_detected<=y_truth+deviation) and (y_detected>=y_truth-deviation):
                        if class_detected[j]=='rojo':
                            row=0                    
                        elif class_detected[j]=='amarillo':
                            row=1
                        elif class_detected[j]=='verde':
                            row=2
                        elif class_detected[j]=='ceda':
                            row=3
                        elif class_detected[j]=='cruce':
                            row=4
                        elif class_detected[j]=='cebra':
                            row=5
                        elif class_detected[j]=='pare':
                            row=6
                        elif class_detected[j]=='peaton':
                            row=7
                        #elif class_detected[j]=='vehiculo':
                            #row=8
                        else:
                            row=9
                        if (row<=8 and row>=0):
                            confusion_m[row,column]=confusion_m[row,column]+1
                deviation=0.05
                #Loop for ideal matrix
                for j in range(len(class_ref)):
                    x_ref=box_ref_x[j]
                    y_ref=box_ref_y[j]
                    if (x_ref<=x_truth+deviation) and (x_ref>=x_truth-deviation) and (y_ref<=y_truth+deviation) and (y_ref>=y_truth-deviation):
                        if class_ref[j]=='rojo':
                            row=0
                        elif class_ref[j]=='amarillo':
                            row=1
                        elif class_ref[j]=='verde':
                            row=2
                        elif class_ref[j]=='ceda':
                            row=3
                        elif class_ref[j]=='cruce':
                            row=4
                        elif class_ref[j]=='cebra':
                            row=5
                        elif class_ref[j]=='pare':
                            row=6
                        elif class_ref[j]=='peaton':
                            row=7
                        #elif class_ref[j]=='vehiculo':
                            #row=8
                        else:
                            row=9
                        if (row<=8 and row>=0):
                            confusion_ref[row,column]=confusion_ref[row,column]+1
                
                #Loop for True Negatives row in Confusion Matrix
                column_sum=0
                for j in range(8):
                    for i in range(8):
                        column_sum=column_sum+confusion_m[i,j]
                    
                    confusion_m[8,j]=confusion_ref[j,j]-column_sum
                    column_sum=0
                #Loop for Confusion normalized matrix
                for i in range(8):
                    for j in range(8):
                            diag=confusion_ref[i,i]
                            confusion_norm[i,j]=confusion_m[i,j]/diag
                            confusion_norm[8,i]=confusion_m[8,i]/diag
    print(confusion_ref)
    print(confusion_m)
    print(confusion_norm)
    class_names=['Rojo','Amarillo','Verde','Ceda','Cruce','Cebra','Pare','Peaton']
    plt.figure()
    plot_confusion_matrix(confusion_norm, classes=class_names,title='Confusion matrix')


def plot_confusion_matrix(cm, classes,
                         title='Confusion Matrix',
                          cmap=plt.cm.PuBuGn):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.1f' # '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted labels')
    plt.xlabel('True labels')
