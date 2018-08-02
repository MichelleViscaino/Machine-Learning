# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:42:46 2018

@author: Leonardo
"""

import pandas as pd
import numpy as np


def speed_extractionv3(frame_init,frame_end,N,folder,file):
    df = pd.read_excel('C:\\tensorflow1\\models\\research\\object_detection\\test_images\\'+folder+file+'.xls', sheet_name=0) # can also index sheet by name or fetch all sheets
    speed_buffer = df.values  
    index=int(len(speed_buffer)/N)
    init=frame_init*index
    #print("inicio: %d" %init)
    end=frame_end*index
    #print("fin:%d" %end)
    speed_buffer = speed_buffer[init:end]
    time=np.linspace(0,1,len(speed_buffer)).reshape(-1,1)
    speed_buffer_2=np.hstack([time,speed_buffer])
    #speed_buffer_3=np.transpose(speed_buffer_2)
    return speed_buffer_2