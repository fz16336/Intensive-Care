# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:36:57 2018

@author: Farrel
"""

import os 
import pandas as pd

dirname = 'C:\\Users\Farrel\Documents\MDM\set-a'
os.chdir(dirname)
#print(os.listdir(dirname))

for filename in os.listdir(dirname):
    #pdata = pd.read_csv(filename)
    print(os.listdir(dirname)[0])
    

