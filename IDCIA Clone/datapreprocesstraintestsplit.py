# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:31:32 2023

@author: asympson
"""

        
import random
import shutil
import os
import torch
import torchvision

source = r'C:\Users\asympson\Documents\IDCIA v2\images'
dest = r'C:\Users\asympson\Documents\IDCIA v2\train'
files = os.listdir(source)
no_of_files = 175

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)

source = r'C:\Users\asympson\Documents\IDCIA v2\images'
dest = r'C:\Users\asympson\Documents\IDCIA v2\test'
files = os.listdir(source)
no_of_files = 50

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
    
    
source = r'C:\Users\asympson\Documents\IDCIA v2\images'
dest = r'C:\Users\asympson\Documents\IDCIA v2\val'
files = os.listdir(source)
no_of_files = 25

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)