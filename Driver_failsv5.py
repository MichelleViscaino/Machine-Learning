# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:37:30 2018

@author: Leonardo
"""

def Driver_failsv5(speed_change,true_det):
    aux=range(len(true_det))
    fail='none'
    alert_pedestrian='off'
    aux_true_det='none' #initial value 
    #first search for a pedestrian in the scene and
    #to give priority in case of a lot of signals detected
    for i in aux: 
        if true_det[i]=='peaton':
            alert_pedestrian='on'
        #to give more priority to semaforo and pare than cruce,cebra and ceda
        if aux_true_det=='pare' or aux_true_det=='rojo': 
            aux_true_det='pare'
        elif aux_true_det=='amarillo' or aux_true_det=='verde':
            aux_true_det='verde'
        else:
            aux_true_det=true_det[i]
    #Inference Rules 
    if aux_true_det=='pare' or aux_true_det=='rojo':
        if speed_change=='detiene':
            fail='t_verde'
        else:
            fail='t_roja'
    elif aux_true_det=='cebra' or aux_true_det=='cruce' or aux_true_det=='ceda':
        if alert_pedestrian=='on':
            if speed_change=='detiene':
                fail='t_verde'
            else:
                fail='t_roja'
        else:
             if speed_change!='aumenta':
                 fail='t_verde'
             else:
                 fail='t_roja'
    elif aux_true_det=='verde' or aux_true_det=='amarillo':
        fail='t_verde'

    return fail