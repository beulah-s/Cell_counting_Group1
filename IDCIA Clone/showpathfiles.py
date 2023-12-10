# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:59:55 2023

@author: asympson
"""

# import OS module
import os
 
# Get the list of all files and directories
path = r"C:\Users\asympson\Documents\IDCIA v2\images"
dir_list = os.listdir(path)
 
print("Files and directories in '", path, "' :")
 
# prints all files
print(dir_list)