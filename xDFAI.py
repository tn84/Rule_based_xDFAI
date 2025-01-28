from tqdm.notebook import tqdm
import copy
import cv2
import json
import numpy as np
import pickle
import torch
import torch.nn as nn
from Function import std_2
import Imf_Extraction ImfExtrat
from Trace_Detection import N_top_func
from Trace_Detection import N_top_indices
from Trace_Value_Extraction import train_top_value
from Trace_Value_Extraction import test_top_value
from NED_SED import NED_SED
from Majority_Voting_Rules import MV
from Majority_Voting_Rules import Rules


class_name = "List of the classes of the dataset"  #e.g. ['1','2','3']
model_layers = "List of your model layers" # e.g. layers=['conv1','fc1']
layers_conv = "List of your model convolution layers" # e.g. layers=['conv1','conv2','conv3','conv4']
model_name = "Name of your model function" #string
class_num = "number of classes of your model" #(integer)
TP=0
TN=0
FP=0
FN=0
Fix=0
Detect=0
Wrong=0
faults_result=[]
fault=0
all_result = []
model_threshold = 4 # Set the threshold based on the formula provided in the paper
N_top_count = "Must be defined based on the formula of paper"

ImfExtrat()
for layer in model_layers:
 for c in class_name:
  N_tops, shap_layer = N_top_func(layer,c)
  sorted_shap_top_indexes, shap_top_value, common_indices_flats = N_top_indices(layer,N_top_count)
   activation_top_value = train_top_value(layer,c)
   test_top_value = test_top_value(layer,'TEST DIR')
   Std_top_value = std_2()
   NED_SED(c, 'TEST Activation DIR')

# Majority Voting
MV()

# Detection based on the rules
Rules()

   
# Explations based on the rules

# with open('Outputs/Test_RuleExtraction1-7.pickle', 'rb') as file:
#     g_all_result = pickle.load(file)

for sublist in g_all_result:
    line = ' '.join(str(item) for item in sublist)
    print(line)
sum_TP = 0
sum_FP = 0
sum_TN = 0
sum_FN = 0
sum_Fix = 0

for sublist in g_all_result:
  sum_TP += sublist[3]
  sum_Fix += sublist[5]
  sum_FN += sublist[7]
  sum_FP += sublist[9]
  sum_TN += sublist[11]

print("TP:",sum_TP,"Accuracy:",(sum_TP/1350)*100,((sum_TP+sum_TN+sum_Fix)/1350)*100)
print("FP:",sum_FP,"Accuracy:",(sum_FP/sum_FP+sum_TN)*100)
print("TN:",sum_TN,"Accuracy:",(sum_TN/sum_FP+sum_TN)*100)
print("FN:",sum_FN,"Accuracy:",(sum_FN/1313)*100)
print("Fix:",sum_Fix,"Accuracy:",(sum_Fix/1350)*100)
print("Sum:",sum_TP+sum_FP+sum_TN+sum_FN)
for item in FP_result:
    print(item)
print("===============================")
for item in FN_result:
    print(item)
