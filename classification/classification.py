#!/usr/bin/python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import pandas as pd
import numpy as np
import time
import json


# In[2]:


from torch import nn
from torch.nn import BCELoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# In[3]:


#開頭分類器
layer = {'Liver parenchyma' : '1_1', 'Liver lesion' : '1_2', 'Liver mass invasion' : '1_3',
         'Liver vessel' : '1_4', 'Ascites' : '1_5', 'Coronary vein' : '1_6',
         'Splenorenal shunt' : '1_7', 'Peritoneal carcinomatosis' : '1_8',
         'Pancreatic parenchyma' : '2_1', 'Pancreatic mass' : '2_2', 'Pancreatic cyst' : '2_3',
         'Visible LNs Presence Localization' : '3_1', 'Abnormal enlarged LN' : '3_2',
         'Spleen' : '4_1', 'Spleen mass' : '4_2', 'Spleen trauma' : '4_3', 
         'Spleen abscess Fluid collection in spleen parenchym' : '4_4',
         'Hydronephrosis' : '5_1', 'Renal stone' : '5_2',
         'Renal tumor' : '5_3', 'Prostate' : '5_4',
         'Adrenal gland' : '5_5', 'Renal calcification Presence' : '5_6',
         'Renal cyst Side' : '5_7',
         'Lung nodule' : '6_1', 'Pericardial effusion' : '6_2',
         'Pleural effusion' : '6_3', 'Calcified Coronary artery plaque Presence' : '01',
         'Atelectasis' : '6_4', 'Consolidation' : '6_5', 'Infiltration' : '6_6',
         'GI tract Unremarkable' : '02', 'GI tract Dilatation/Distention' : '7_1',
         'GI tract Ischemia' : '7_2', 'GI tract Free air' : '7_3', 'GI tract Mass' : '7_4',
         'GI tract Right Inguinal hernia' : '7_5', 'GI tract Left Inguinal hernia' : '7_6',
         'GI tract Bil. Inguinal hernia' : '7_7', 'GI tract Umbilical hernia' : '7_8',
         'GI tract Ventral hernia' : '7_9', 'GI tract Appendix' : '7_10',
         'Bone Degeneration' : '03', 'Bone Unremarkable' : '04',
         'Bone Osteolytic Lesion' : '8_1', 'Bone Osteblastic Lesion' : '8_2',
         'Gallbladder' : '9_1', 'Bile duct' : '9_2', 'Aorta Unremarkable' : '05',
         'Aorta Atherosclerotic' : '06', 'Aorta Aneurysm' : '10_1',
         'Liver transplant Liver parenchyma' : '11_1', 'Liver transplant Hepatic artery' : '11_2',
         'Liver transplant Hepati vein' : '11_3', 'Liver transplant Portal vein' : '11_4',
         'Liver transplant Liver volume' : '11_5'}


# In[4]:


#Liver分類器         
layer_1_1 = {'Remarkable Size' : '1' , 'Remarkable Cirrhosis' : '2',
             'Remarkable Diffuse change' : '3', 'Remarkable Post theraputic change' : '4',
             'Remarkable Viable lesion' : '5', 'Unremarkable' : '01'}
layer_1_1_1 = {'Hypertrophy' : '01', 'Atrophy' : '02'}
layer_1_1_2 = {'No' : '01', 'Yes' : '02'}
layer_1_1_3 = {'Not presence' : '01', 'Low attenuation' : '02', 'Hign attenuation' : '03'}
layer_1_1_4 = {'Post Resection' : '1', 'Post TAE/TACE' : '2', 
               'Post Pigtail drainage' : '3', 'Post RFA' : '4', 
               'Post ethanol injection' : '5', 'Post Y90 radioembolization' : '6'}
layer_1_1_4_1 = {'S/P S1-8' : '1', 'S/P Right hepatectomy' : '01',
                 'S/P Left hepatectomy' : '02',}
layer_1_1_4_1_1 = {'S/P S1' : '01', 'S/P S2' : '02', 'S/P S3' : '03', 'S/P S4' : '04',
                   'S/P S5' : '05', 'S/P S6' : '06', 'S/P S7' : '07', 'S/P S8' : '08'}
layer_1_1_4_2 = {'With lipiodol and doxorubicin' : '1', 'With lipiodol stasis' : '2',
                 'With tendem' : '3', 'With hepasphere' : '4', 'With DC bead' : '5'}
layer_1_1_4_2_1 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                   'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                   'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                   'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                   'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                   'S5/8 Size[mm]' : '016'}
layer_1_1_4_2_2 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                   'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                   'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                   'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                   'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                   'S5/8 Size[mm]' : '016'}
layer_1_1_4_2_3 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                   'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                   'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                   'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                   'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                   'S5/8 Size[mm]' : '016'}
layer_1_1_4_2_4 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                   'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                   'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                   'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                   'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                   'S5/8 Size[mm]' : '016'}
layer_1_1_4_2_5 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                   'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                   'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                   'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                   'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                   'S5/8 Size[mm]' : '016'}
layer_1_1_4_3 = {'PTGBD' : '01', 'PTCD' : '02', 'Abscess' : '03'}
layer_1_1_4_4 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_1_4_5 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_1_4_6 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_1_5 = {'S1' : '01', 'S2' : '02', 'S3' : '03', 'S4' : '04',
               'S5' : '05', 'S6' : '06', 'S7' : '07', 'S8' : '08'}
layer_1_2 = {'Single lesion enhancement' : '1',
             'Multiple lesion enhancement' : '2', 'Not presence' : '01'}
layer_1_2_1 = {'Delay enhancement' : '1', 'Poor enhancement' : '2',
               'Mild enhancement' : '3', 'Peripheral enhancement' : '4',
               'Hypodense favoring cysts Localization' : '5',
               'Early enhancement, early washout favoring HCC Localization' : '6',
               'No early enhancement but early washout favoring cysts Localization' : '7',
               'Early enhancement possible hemangomas Localization' : '8',
               'Early enhancement, persistent enhancement possible hemangomas Localization' : '9',
               'Early enhancement,to isodensity possible hemangomas Localization' : '10',
               'Heterogenouse enhancement possible HCC Localization' : '11',
               'No definite enhancement possible cysts or other cause Localization' : '12',
               'Peripheral to centripetal enhancement possible hemangomas Localization' : '13'}
layer_1_2_1_1 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_1_1_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_1_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_1_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_1_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_1_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_1_2_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_2_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_2_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_2_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_2_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_1_3_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_3_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_3_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_3_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_3_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4 = {'Possible abscess Localization' : '1', 'possible hemangomas' : '2'}
layer_1_2_1_4_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_4_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_4_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_4_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_5 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_2_1_6 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_2_1_7 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_7_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_7_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_8_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_8_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_9_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_9_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_10_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_10_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_11_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_11_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_12_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_12_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_1_13_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_1_13_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2 = {'Delay enhancement' : '1', 'Poor enhancement' : '2',
               'Mild enhancement' : '3', 'Peripheral enhancement' : '4',
               'Hypodense favoring cysts Localization' : '5',
               'Early enhancement, early washout favoring HCC Localization' : '6',
               'No early enhancement but early washout favoring cysts Localization' : '7',
               'Early enhancement possible hemangomas Localization' : '8',
               'Early enhancement, persistent enhancement possible hemangomas Localization' : '9',
               'Early enhancement,to isodensity possible hemangomas Localization' : '10',
               'Heterogenouse enhancement possible HCC Localization' : '11',
               'No definite enhancement possible cysts or other cause Localization' : '12',
               'Peripheral to centripetal enhancement possible hemangomas Localization' : '13'}
layer_1_2_2_1 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_2_1_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_1_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_1_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_1_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_1_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_2_2_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_2_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_2_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_2_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_2_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3 = {'possible HCC Localization' : '1',
                 'possible cholangiocarconoma Localization' : '2',
                 'Possible metastasis Localization' : '3'}
layer_1_2_2_3_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_3_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_3_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_3_3_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_3_3_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4 = {'Possible abscess Localization' : '1', 'possible hemangomas' : '2'}
layer_1_2_2_4_1 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_4_1_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_1_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_4_2_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_4_2_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_5 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_2_2_6 = {'S1 Size[mm]' : '01', 'S2 Size[mm]' : 'E02nd_2', 'S3 Size[mm]' : '03',
                 'S4 Size[mm]' : '04', 'S5 Size[mm]' : '05', 'S6 Size[mm]' : '06',
                 'S7 Size[mm]' : '07', 'S8 Size[mm]' : '08', 'S4/8 Size[mm]' : '09',
                 'S4/5 Size[mm]' : '010', 'S7/8 Size[mm]' : '011', 'S5/6 Size[mm]' : '012',
                 'S6/7 Size[mm]' : '013', 'S2/4 Size[mm]' : '014', 'S2/3 Size[mm]' : '015',
                 'S5/8 Size[mm]' : '016'}
layer_1_2_2_7 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_7_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_7_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_8_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_8_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_9_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_9_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_10_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_10_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_11_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_11_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_12_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_12_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13 = {'S1' : '1', 'S2' : '2', 'S3' : '3', 'S4' : '4', 'S5' : '5', 
                   'S6' : '6', 'S7' : '7', 'S8' : '8', 'S4/8' : '9', 'S4/5' : '10',
                   'S7/8' : '11', 'S5/6' : '12', 'S6/7' : '13', 'S2/4' : '14',
                   'S2/3' : '15', 'S5/8' : '16'}
layer_1_2_2_13_1 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_2 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_3 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_4 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_5 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_6 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_7 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_8 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_9 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_10 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_11 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_12 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_13 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_14 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_15 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_2_2_13_16 = {'Size[mm]' : '01', 'DDX:[freetext]' : '02'}
layer_1_3 = {'Invasion Vascular invasion' : '1',
             'Invasion Bile duct' : '2', 'Invasion Organ invasion' : '3',
             'No Invasion' : '01',}
layer_1_3_1 = {'Right portal vein' : '01', 'Left portal vein' : '02',
               'Main portal vein' : '03', 'Middle hepatic vein' : '04',
               'Left hepatic vein' : '05', 'Right hepatic vein' : '06',
               'Right hepatic artery' : '07', 'Left hepatic artery' : '08', 'IVC' : '09'}
layer_1_3_2 = {'Hilum' : '01', 'Right IHD' : '02', 'Left IHD' : '03'}
layer_1_3_3 = {'Gallbladder' : '01', 'Stomach' : '02', 'Kidney' : '03', 
               'Spleen' : '04', 'Pancreas' : '05'}

layer_1_4 = {'Hepatic artery' : '1', 'Hepatic vein' : '2', 'Portal vein' : '3'}
layer_1_4_1 = {'Unremarkable' : '01', 'Remarkable Stenosis' : '02', 
               'Remarkable Occlusion' : '03', 'Remarkable Dissection' : '04', 
               'Remarkable Aneurysm' : '05'}
layer_1_4_2 = {'Unremarkable' : '01', 'Remarkable Thrombus Right hepatic  vein' : '02',
               'Remarkable Thrombus Middle hepatic vein' : '03',
               'Remarkable Thrombus Left hepatic vein' : '04'}
layer_1_4_3 = {'Unremarkable' : '1', 'Remarkable Thrombus' : '2'}
layer_1_4_3_1 = {'Main portal vein and branches' : '01', 'Main portal vein' : '02',
                'Right portal vein' : '03', 'RAPV' : '04', 'RPPV' : '05',
                'Left portal vein' : '06'}
layer_1_4_3_2 = {'Main portal vein' : '01', 'Right portal vein' : '02', 'RAPV' : '03',
                'RPPV' : '04', 'Left portal vein' : '05'}

layer_1_5 = {'Presence Amount' : '1', 'Presence Location' : '2', 'Not presence' : '01'}
layer_1_5_1 = {'Mild' : '01', 'Moderate' : '02', 'Severe' : '03'}
layer_1_5_2 = {'Perihepatic region' : '01', 'Perisplenic region' : '02',
               'Upper abdomen' : '03', 'Lower abdomen' : '04', 'Pelvic cavity' : '05'}
layer_1_6 = {'Not presence' : '01', 'Presence Size (mm) [1..n]' : '02'}
layer_1_7 = {'Not presence' : '01', 'Presence Size (mm) [1..n]' : '02'}
layer_1_8 = {'Not presence' : '01', 'Presence' : '02'}


# In[5]:


#Pancreas分類器
layer_2_1 = {'Remarkable Parenchyma' : '1',
             'Remarkable Peripancreatic fluid' : '2', 
             'Unremarkable' : '01', 'Remarkable Edema' : '02'}
layer_2_1_1 = {'Calcification' : '1','Atrophy' : '01', 'Fat infiltration' : '02',
               'Duct dilataiton Size[mm][1..n]' : '03'}
layer_2_1_1_1 = {'Signs of chronic pancreatitis' : '01', 'Cause unknown' : '02'}
layer_2_1_2 = {'No' : '01', 'Yes Peripancreatic' : '02',
               'Yes Right anterior pararenal space' : '03',
               'Yes Left anterior pararenal space' : '04',
               'Yes Retroperitoneal region' : '05', 'Yes Lesser sac' : '06'}
layer_2_2 = {'Yes Location' : '1', 'Yes Enhancement' : '2',
             'Yes Adjacent  invasion' : '3', 'No' : '01', 'Yes Size[mm]' : '02',
             'Yes Ill defined' : '03', 'Yes Series/number [SE/IM]' : '04'}
layer_2_2_1 = {'Pancreatic head' : '01', 'Pancreatic body' : '02',
               'Pancreatic tail' : '03', 'Pancreatic uncinate process' : '04'}
layer_2_2_2 = {'Poor enhanced' : '01', 'Strongly enhanced' : '02',
               'Isodense' : '03'}
layer_2_2_3 = {'Organ' : '1', 'Vascular structure' : '2'}
layer_2_2_3_1 = {'Liver' : '01', 'Gallbladder' : '02', 'Bile duct' : '03',
                 'Right kidney' : '04', 'Left kidney' : '05', 'Stomach' : '06',
                 'Spleen' : '07', 'Bone' : '08'}
layer_2_2_3_2 = {'IVC' : '01', 'Celiac trunk' : '02', 'SMA' : '03'}
layer_2_3 = {'Yes Location' : '1', 'No' : '01'}
layer_2_3_1 = {'Head Size[mm][1..n]' : '01', 'body Size[mm][1..n]' : '02', 
               'tail Size[mm][1..n]' : '03', 'Uncinate process Size[mm][1..n]' : '04'}


# In[6]:


#Lymph node分類器
layer_3_1 = {'Retroperitoneum' : '01', 'Mesentary' : '02', 'Pelvic cavity' : '03'}
layer_3_2 = {'Presence Localization' : '1',
             'Presence Confidence in malignancy' : '2', 'Not presence' : '01'}
layer_3_2_1 = {'Retrocrural' : '1', 'Gastrohepatic' : '2',
               'Celiac' : '3', 'SMA' : '4', 'IMA' : '5', 'Paraaortic' : '6',
               'Aortalcaval' : '7', 'Paracaval' : '8', 'Porta hepatis' : '9',
               'Peri-pancreatic' : '10', 'Peri-splenic' : '11',
               'Great curvature' : '12', 'Lesser curvature' : '13', 
               'Peri-colonal' : '14', 'Cardiophrenic' : '15', 'ommon iliac' : '16', 
               'External iliac' : '17', 'Internal iliac' : '18', 'Obturator' : '19'}

layer_3_2_1_1 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_2 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_3 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_4 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_5 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_6 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_7 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_8 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_9 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_10 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_11 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_12 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_13 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_14 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_15 = {'Right' : '1', 'Left' : '2'}
layer_3_2_1_15_1 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_15_2 = {'Size[mm][1..n]' : '01', 'Number [1..n]' : '02'}
layer_3_2_1_16 = {'Right Size[mm][1..n]' : '01', 'Left Size[mm][1..n]' : '02'}
layer_3_2_1_17 = {'Right Size[mm][1..n]' : '01', 'Left Size[mm][1..n]' : '02'}
layer_3_2_1_18 = {'Right Size[mm][1..n]' : '01', 'Left Size[mm][1..n]' : '02'}
layer_3_2_1_19 = {'Right Size[mm][1..n]' : '01', 'Left Size[mm][1..n]' : '02'}
layer_3_2_2 = {'Possible reactive change' : '01', 'Unlikely for (<10%)' : '02',
               'Less liklely for (~25%)' : '03', 'Possible for (~50%)' : '04',
               'Suspecious for (~75%)' : '05', 'Consistent with(~90%)' : '06'}


# In[7]:


#Spleen分類器
layer_4_1 = {'Remarkable Splenomegaly' : '1', 'Unremarkable' : '01'}
layer_4_1_1 = {'Mild' : '01', 'Moderate' : '02', 'Marked' : '03'}
layer_4_2 = {'Non-enhanced lesion' : '1', 'Enhanced lesion' : '2',
             'Ill defined lesion' : '3', 'None' : '01'}
layer_4_2_1 = {'Maybe cyst Size[mm][1..n]' : '01',
               'Nature to be determined Size[mm][1..n]' : '02'}
layer_4_2_2 = {'Maybe hemangioma Size[mm][1..n]' : '01',
               'Nature to be determined Size[mm][1..n]' : '02'}
layer_4_2_3 = {'Suspect metastasis' : '01', 'Nature to be determined' : '02',
               'Suspect lymphoma' : '03'}
layer_4_3 = {'Subcapsular hematoma' : '1', 'Intraparenchymal hematoma' : '2',
             'Parenchymal laceration' : '3', 'Vascular injury' : '4',
             'Active bleeding' : '5', 'Shattered spleen' : '6',
             'hematoma Rupture subcapsular or parenchymal hematoma' : '01'}
layer_4_3_1 = {'< 10% surface area' : '01', '10-50% surface area' : '02',
               '> 50% surface area' : '03'}
layer_4_3_2 = {'< 5cm' : '01', '≥ 5 cm' : '02',}
layer_4_3_3 = {'Capsular < 1cm depth' : '01', '1-3cm in depth' : '02',
               '> 3cm in depth or involving vessel' : '03',
               'parenchymal laceration involving segmental or hilar vessels producing >25% devascularisation' : '04'}
layer_4_3_4 = {'Not presence' : '01', 'Presence' : '02'}
layer_4_3_5 = {'Not presence' : '01', 'Presence' : '02'}
layer_4_3_6 = {'Not presence' : '01', 'Presence' : '02'}
layer_4_4 = {'Not presence' : '01', 'Presence Fluid' : '02', 'Presence Blood' : '03',
             'Presence Fluid and Air' : '04'}


# In[8]:


#Kidney分類器
layer_5_1 = {'Prescence Right' : '1', 'Prescence Left' : '2',
             'Prescence Both' : '3', 'Not presence' : '01'}
layer_5_1_1 = {'Mild renal pelvis dilatation without involvement of calyces' : '01',
               'Moderate renal pelvis dilatation with involvement of some calyces' : '02',
               'Severe renal pelvis dilatation with parenchymal thinning and uniform dilatation of all calyces' : '30'}
layer_5_1_2 = {'Mild renal pelvis dilatation without involvement of calyces' : '01',
               'Moderate renal pelvis dilatation with involvement of some calyces' : '02',
               'Severe renal pelvis dilatation with parenchymal thinning and uniform dilatation of all calyces' : '30'}
layer_5_1_3 = {'Mild renal pelvis dilatation without involvement of calyces' : '01',
               'Moderate renal pelvis dilatation with involvement of some calyces' : '02',
               'Severe renal pelvis dilatation with parenchymal thinning and uniform dilatation of all calyces' : '30'}
layer_5_2 = {'Calyx' : '1', 'Side' : '2'}
layer_5_2_1 = {'Upper Size[mm][1..n]' : '01', 'Middle Size[mm][1..n]' : '02',
               'Lower' : '03', 'Staghorn' : '04'}
layer_5_2_2 = {'Right' : '01', 'Left' : '02'}
layer_5_3 = {'Contour' : '1', 'Pattern' : '2', 'Invasion' : '3', 'Side' : '4'}
layer_5_3_1 = {'Well defined' : '01', 'ill defined' : '02'}
layer_5_3_2 = {'Enhancement' : '01', 'Poor enhancement' : '02', 'Fat content' : '03'}
layer_5_3_3 = {'IVC' : '01', 'Colon' : '02', 'Pancreas' : '03', 'Stomach' : '04',
               'Liver' : '05', 'Aorta' : '06', 'Renal vein' : '07', 'Renal artery' : '08'}
layer_5_3_4 = {'Right' : '01', 'Left' : '02'}
layer_5_4 = {'Enlarged' : '1', 'Post OP' : '01'}
layer_5_4_1 = {'Calcification' : '01', 'No calcification' : '02'}
layer_5_5 = {'Right' : '1', 'Left' : '2'}
layer_5_5_1 = {'Nodule' : '1', 'Thickening' : '01'}
layer_5_5_1_1 = {'Hypodense Size[mm][1..n]' : '01', 'Enhanced Size[mm][1..n]' : '02'}
layer_5_5_2 = {'Nodule' : '1', 'Thickening' : '01'}
layer_5_5_2_1 = {'Hypodense Size[mm][1..n]' : '01', 'Enhanced Size[mm][1..n]' : '02'}
layer_5_6 = {'Right' : '01', 'Left' : '02'}
layer_5_7 = {'Right Size[mm][1..n]' : '01', 'Left Size[mm][1..n]' : '02'}


# In[9]:


#Chest分類器
layer_6_1 = {'Multiple nodules' : '1', 'Single nodule' : '2', 'Not presence' : '01'}
layer_6_1_1 = {'Side' : '1', 'Pattern' : '2', 'Trend' : '3'}
layer_6_1_1_1 = {'Right lower' : '01', 'Right middle' : '02', 'Ingular lung' : '03',
                 'Left lower' : '04'}
layer_6_1_1_2 = {'Solid' : '01', 'Ground glass' : '02', 'Calcified' : '03',
                 'Subsolid' : '04'}
layer_6_1_1_3 = {'Progression change' : '01', 'Stable appearance' : '02',
                 'Regression change' : '03', 'Newly occurred' : '04'}
layer_6_1_2 = {'Side' : '1', 'Pattern' : '2', 'Trend' : '3',
               'Size[mm][1..n]' : '01', 'Series/number [SE/IM]' : '02'}
layer_6_1_2_1 = {'Right lower' : '01', 'Right middle' : '02', 'Ingular lung' : '03',
                 'Left lower' : '04'}
layer_6_1_2_2 = {'Solid' : '01', 'Ground glass' : '02', 'Calcified' : '03',
                 'Subsolid' : '04'}
layer_6_1_2_3 = {'Progression change' : '01', 'Stable appearance' : '02',
                 'Regression change' : '03', 'Newly occurred' : '04'}
layer_6_2 = {'Not presence' : '01', 'Presence Max thickness (mm) [1..n]' : '02'}
layer_6_3 = {'Not presence' : '01', 'Presence Right' : '02', 'Presence left' : '03'}
layer_6_4 = {'Right middle lung' : '01', 'Right lower lung' : '02',
             'Lingular lung' : '03', 'Left lower lung' : '04'}
layer_6_5 = {'Right middle lung' : '01', 'Right lower lung' : '02',
             'Lingular lung' : '03', 'Left lower lung' : '04'}
layer_6_6 = {'Right middle lung' : '01', 'Right lower lung' : '02',
             'Lingular lung' : '03', 'Left lower lung' : '04'}


# In[10]:


#GI tract分類器
layer_7_1 = {'Small bowel' : '01', 'Colon' : '02', 'Stomach' : '03'}
layer_7_2 = {'Yes Small bowel' : '1', 'None' : '01', 'Yes Colon' : '02'}
layer_7_2_1 = {'Duodenum' : '01', 'Jejunum' : '02', 'Ileum' : '03'}
layer_7_3 = {'None' : '01', 'Yes' : '02'}
layer_7_4 = {'Yes Stomach' : '1', 'Yes Colon' : '2', 'Yes Small bowel' : '3', 
             'Yes Adjacent  invasion' : '4', 'None' : '01'}
layer_7_4_1 = {'Body' : '1', 'Fundus' : '2', 'Cardica' : '3', 'Antrum' : '4'}
layer_7_4_1_1 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_1_2 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_1_3 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_1_4 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2 = {'Cecum' : '1', 'A-colon' : '2', 'Hepatic flexure' : '3',
               'T-colon' : '4', 'Splenic flexure' : '5', 'D-colon' : '6',
               'S-colon' : '7', 'Rectum' : '8'}
layer_7_4_2_1 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_2 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_3 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_4 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_5 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_6 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_7 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_2_8 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_3 = {'Duodenum' : '1', 'Jejunum' : '2', 'Ileum' : '3'}
layer_7_4_3_1 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_3_2 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_3_3 = {'Enhanced' : '01', 'Non-Enhanced' : '02'}
layer_7_4_4 = {'organ' : '1', 'Vascular structure' : '2'}
layer_7_4_4_1 = {'Liver' : '01', 'Gallbladder' : '02', 'Bile duct' : '03',
                 'Right kidney' : '04', 'Left kidney' : '05', 'Stomach' : '06', 
                 'Spleen' : '07', 'Bone' : '08'}
layer_7_4_4_2 = {'IVC' : '01', 'Celiac trunk' : '02', 'SMA duct' : '03'}
layer_7_5 = {'Bowel loop incaceration' : '1', 'Mesenteric fat' : '01'}
layer_7_5_1 = {'No' : '01', 'Yes' : '02'}
layer_7_6 = {'Bowel loop incaceration' : '1', 'Mesenteric fat' : '01'}
layer_7_6_1 = {'No' : '01', 'Yes' : '02'}
layer_7_7 = {'Bowel loop incaceration' : '1', 'Mesenteric fat' : '01'}
layer_7_7_1 = {'No' : '01', 'Yes' : '02'}
layer_7_8 = {'Bowel loop incaceration' : '1', 'Mesenteric fat' : '01'}
layer_7_8_1 = {'No' : '01', 'Yes' : '02'}
layer_7_9 = {'Bowel loop incaceration' : '1', 'Mesenteric fat' : '01'}
layer_7_9_1 = {'No' : '01', 'Yes' : '02'}
layer_7_10 = {'Not dilated' : '01', 'Dilated' : '02', 'Dilated Appendicolith' : '03',
              'Dilated Periappendical stranding' : '04',
              'Dilated Periappendical fluid' : '05', 'Dilated Mass' : '06'}


# In[11]:


#Bone分類器


# In[12]:


#Gallbladder and bile duct分類器
layer_9_1 = {'Remarkable' : '1', 'Unremarkable' : '01', 'Cholecystectomy' : '02'}
layer_9_1_1 = {'Wall thickening' : '1', 'Stone' : '01', 'Sludg' : '02'}
layer_9_1_1_1 = {'Pericholecystic stranding' : '1', 'None' : '01'}
layer_9_1_1_1_1 = {'None' : '01', 'Yes' : '02'}
layer_9_2 = {'Dilatation' : '1', 'Stone' : '2', 'Pneumobilia' : '3', 'ERBD' : '4'}
layer_9_2_1 = {'None' : '01', 'CBD Size[mm][1..n]' : '02', 
               'Right IHD Size[mm][1..n]' : '03', 'Left IHD Size[mm][1..n]' : '04'}
layer_9_2_2 = {'None' : '01', 'CBD' : '02', 'Right IHD' : '03', 'Left IHD' : '04',
               'CHD' : '05', 'Cystic duct' : '06'}
layer_9_2_3 = {'Right IHD' : '01', 'Left IHD' : '02'}
layer_9_2_4 = {'Right IHD' : '01', 'Left IHD' : '02', 'CHD' : '03'}


# In[13]:


#Liver transplant分類器
layer_11_1 = {'Cirrhosis' : '1',
              'Low attenuation Fatty liver percentage % [percentage]' : '01',
              'Liver HU number (HU) [Hu]' : '02', 'Spleen HU number (HU) [Hu]' : '03'}
layer_11_1_1 = {'No' : '01', 'Yes' : '02', 'Unremarkable' : '03'}
layer_11_2 = {'RHA' : '1', 'LHA' : '2', 'S4HA' : '3'}
layer_11_2_1 = {'From proper HA' : '01', 'From SMA' : '02'}
layer_11_2_2 = {'From proper HA' : '01', 'From LGA' : '02', 'From SMA' : '03'}
layer_11_2_3 = {'From LHA' : '01', 'From RHA ' : '02'}
layer_11_3 = {'MHV and LHV' : '1', 'S8V' : '2',
              'Single right hepatic vein Size[mm][1..n]' : '01', 
              'S2V Size[mm][1..n]' : '02', 'S3V Size[mm][1..n]' : '03',
              'IRHV Size[mm][1..n]' : '04'}
layer_11_3_1 = {'Common' : '1', 'Seperated ' : '2'}
layer_11_3_1_1 = {'MHV size' : '01', 'LHV size ' : '02'}
layer_11_3_1_2 = {'MHV size' : '01', 'LHV size ' : '02'}
layer_11_3_2 = {'Drain into MHV Size[mm][1..n]' : '01',
                'Drain into IVC Size[mm][1..n] ' : '02'}
layer_11_4 = {'Normal configuration' : '1', 'Trifurcation ' : '2'}
layer_11_4_1 = {'MPV Size[mm][1..n]' : '01', 'LPV Size[mm][1..n]' : '02',
                'RPV Size[mm][1..n]' : '03'}
layer_11_4_2 = {'RAPV Size[mm][1..n]' : '01', 'RPPV Size[mm][1..n]' : '02',
                'MPV Size[mm][1..n]' : '03', 'LPV Size[mm][1..n]' : '04'}
layer_11_5 = {'Right liver Volume (cm3) [volume]' : '01', 
              'LLS liver Volume (cm3) [volume]' : '02',
              'S4 liver Volume (cm3) [volume]' : '03',
              'Spleen Volume (cm3) [volume]' : '04'}


# In[14]:


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    '''
    This custom class closely resembles BertForSequenceClassification, which
    supports multiclass classification, but not multi-label.
    This modified version supports data points with multiple labels.
    '''

    def __init__(self, config):
        '''
        Class initializer, called when we create a new instance of this class.
        '''

        # Call the init function of the parent class (BertPreTrainedModel)        
        super().__init__(config)
       
        # Store the number of labels.
        self.num_labels = config.num_labels
        
        # Create a `BertModel`--this implements all of BERT except for the final
        # task-specific output layer (which is what we'll do here in `forward`). 
        self.bert = BertModel(config)

        # Setup dropout object (note: I'm not familiar enough to speak to this).
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Create a [768 x 6] weight matrix to use as our classifier.
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        #self.layer1 = nn.Linear(config.hidden_size, 603)
        #self.layer2 = nn.Linear(603, 438)
        #self.layer3 = nn.Linear(438, config.num_labels)
        #self.relu = nn.ReLU()

        self.Sigmoid = nn.Sigmoid()

        # Initialize model weights (inherited function).
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        '''
        This function defines what happens on a forward pass of our model, both
        for training and evaluation. For example, when we call 
            `model(b_input_ids, ...)`
        during our training loop, it results in a call to this `forward`
        function.
        '''

        # ====================
        #   Run Through BERT
        # ====================

        # All of BERT's (non-task-specific) architecture is implemented by the
        # BertModel class. Here we pass all of the inputs through our BertModel
        # instance. 
        outputs = self.bert(
            input_ids,                      # The input sequence
            attention_mask=attention_mask,  # Mask out any [PAD] tokens.
            token_type_ids=token_type_ids,  # Identify segment A vs. B
            position_ids=position_ids,      # TODO...
            head_mask=head_mask,            # TODO...
            inputs_embeds=inputs_embeds,    # Presumably the initial embeddings
                                            # for the tokens in our sequence.
            output_attentions=output_attentions, # Boolean, whether to return
                                                 # all of the attention scores.
            output_hidden_states=output_hidden_states, # Whether to return
                                                       # embeddings from all 12
                                                       # layers.
        )

        # Side note: It confused me to see us *invoking* an instance of a class
        # (calling self.bert(...)) as if it were a function! I learned that in 
        # Python, an instance of a class can be callable if the class defines a 
        # `__call__` method! 
        # BertModel ultimately inherits from torch.nn.Module, which I imagine 
        # implements a `__call__` method that allows PyTorch to work its magic.

        # The forward pass of 'BertModel' (the call to `self.bert`) returns two
        # items.

        # The first output is the final embeddings taken from the output of 
        # the final BERT encoder layer.
        #
        # `final_embeddings` has dimensions:
        #    [ batch size  x  sequence length  x  768]
        #      (768 is the length of the embeddings in BERT-base)
        #
        # I've included this here for informational purposes, but we won't 
        # actually use the `final_embeddings` anywhere here!
        final_embeddings = outputs[0]

        # ===========================
        #   Apply Output Classifier
        # ===========================

        # The second output is the activated form of the final [CLS] embedding. 
        # This comes from the so-called "pooling layer" that BERT has on its 
        # output which is only applied to the [CLS] token and none of the
        # others.
        #
        # You can see the definition of BertPooler.forward here:
        # https://github.com/huggingface/transformers/blob/0735def8e1200ed45a2c33a075bc1595b12ef56a/src/transformers/modeling_bert.py#L506
        #
        # It takes the final embedding for the [CLS] token (and *only* that
        # token), multiplies it with a [768 x 768] weight matrix, and then
        # applies tanh activation to each of the 768 features in the embedding.
        activated_cls = outputs[1]

        # Apply dropout (note: I'm not familiar enough with dropout to speak to
        # it, but I believe it is applied during training only, and is turned 
        # off during evaluation mode when we call `model.eval()`).
        activated_cls = self.dropout(activated_cls)
        
        # Send it through our linear "classifier". The "classifier" is actually
        # just a [768 x 6] weight matrix, with *no activation function*. 
        # Multiplying the activated CLS embedding with this matrix results in
        # a vector with 6 values, which are the scores for each of our classes.
        # Because we have not applied the activation function, these output 
        # values are referred to as "logits". 
        # When performing evaluation (not training), the logits are adequate for
        # making a classification, since the activation function does not change
        # the ranking of the results.
        # So, in evaluation mode, we are done here!

        logits = self.classifier(activated_cls)

        #logits = self.layer1(self.relu(activated_cls))
        #logits = self.layer2(self.relu(logits))
        #logits = self.layer3(self.relu(logits))

        logits = self.Sigmoid(logits)
        logits = logits.double()

        
        # ===================
        #   Training Mode
        # ===================

        # If labels for the inputs have been provided, we take that to mean that
        # we are in training mode, and we need to calculate the loss function.
        if labels is not None:
            
            # The Binary Cross-Entropy Loss function is defined for us in 
            # PyTorch by the `BCEWithLogitsLoss` class.
            #
            # This loss function will:
            #   1. Apply the sigmoid activation to each of our 6 logit values.
            #   2. Feed those outputs, along with the correct labels, through 
            #      the binary cross entropy loss function to calculate a 
            #      (single?) loss value for the sample.
            #loss_fct = BCEWithLogitsLoss()

            loss_fct = BCELoss()

            # Call the loss function, giving it the `logits` and the correct
            # `labels`.
            loss = loss_fct(logits.view(-1, self.num_labels), # The logits
                            labels.view(-1, self.num_labels)) # The labels

            # What's view(-1, ...)?
            # The `view` function is used to reshape tensors. `-1` tells PyTorch
            # to infer that dimension by dividing the total number of elements
            # by the other dimensions.
            # For batched input, this call to view is not necessary. Both
            # `logits` and `labels` are already [16 x 6] here.
            # Perhaps it's there to re-shape the tensors if you're only
            # evaluating on a single input instead of a batch?

            # Output is (loss, logits, <bonus returns>)
            # The 'bonus return' values are the attentions and the hidden states
            # from all 12 layers, but these are only returned by `BertModel` if
            # the appropriate flags are set. 
            return ((loss, logits) + outputs[2:])

        # ===================
        #   Evaluation Mode
        # ===================

        # Otherwise, in evaluation mode...
        else:
        
            # Output is (logits, <bonus returns>)
            # Again, the logits are adequate for classification, so we don't
            # bother applying the (sigmoid) activation function here.
            return ((logits,) + outputs[2:]) 


# In[15]:


def to_tokenizer(test, test_labels):
    input_ids = []
    attn_masks = []
    labels = []

    # ======== Encoding ========

    print('Encoding all {:,} test samples...'.format(len(test)))

    # For every test sample...
    for (index, row) in test.iterrows():

        # Report progress.
        if ((len(input_ids) % 1000) == 0):
            print('  Tokenized {:,} comments.'.format(len(input_ids)))

        # Convert sentence pairs to input IDs, with attention masks.
        encoded_dict = tokenizer.encode_plus(row['Report_Content'],  # The text to encode.
                                            max_length=max_len,    # Pad or truncate to this lenght.
                                            pad_to_max_length=True,
                                            truncation=True,
                                            add_special_tokens = True,
                                            return_tensors='pt')   # Return objects as PyTorch tensors.

        # Add this example to our lists.
        input_ids.append(encoded_dict['input_ids'])
        attn_masks.append(encoded_dict['attention_mask'])

    print('\nDONE. {:,} examples.'.format(len(input_ids)))


    # ======== List of Examples --> Tensor ========

    # Convert each Python list of Tensors into a 2D Tensor matrix.
    input_ids = torch.cat(input_ids, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)
    
    labels = test_labels.to_numpy().astype(float)

    # Cast the labels list to a 2D Tensor.
    labels = torch.tensor(labels)

    # ======== Summary ========

    print('\nData structure shapes:')
    print('   input_ids:  {:}'.format(str(input_ids.shape)))
    print('  attn_masks:  {:}'.format(str(attn_masks.shape)))
    print('      labels:  {:}'.format(str(labels.shape)))
    return input_ids, attn_masks, labels


# In[16]:


max_len = 128
batch_size = 4


# In[17]:


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[18]:


if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[19]:


def Pred(path, data):
    try:
        data_labels = pd.DataFrame(columns=pd.read_csv(f'{path}/TrainDatas.csv').columns[5:], index = range(1))
    except FileNotFoundError:
        return
    data_labels = data_labels.fillna(0)
    
    input_ids, attn_masks, labels = to_tokenizer(data, data_labels)
    test_dataset = TensorDataset(input_ids, attn_masks, labels)
    test_dataloader = DataLoader(test_dataset,
                                 sampler = SequentialSampler(test_dataset),
                                 batch_size = batch_size)
    flat_predictions, flat_true_labels = eval_model(path, test_dataloader)
    label_cols = data_labels.columns.to_list()
    
    True_index = []
    one_hot = dynamic_thresholds(1, flat_predictions)
    for i in range(len(one_hot[0])):
        if one_hot[0][i] != 0:
            True_index.append(i)
    
    for i, True_idx in enumerate(True_index):
        print('預測為 {} 類別'.format(label_cols[True_idx]))
        if label_cols[True_idx].find('0') != 0:
            if i != 0:
                path = os.path.dirname(path) + '/' + label_cols[True_idx]
            else:
                path += '/' + label_cols[True_idx]
            #print('i', i)
            #print('idx', True_idx)
            #path += '/' + label_cols[True_idx]
            #print(path)
            Pred(path, data)
        else:
            output_label = path.split('/')[2:]
            output_label.append(label_cols[True_idx])
            label.append(output_label)
            print('label', output_label)


# In[20]:


def to_tokenizer(test, test_labels):
    input_ids = []
    attn_masks = []
    labels = []

    # ======== Encoding ========

    print('Encoding all {:,} test samples...'.format(len(test)))

    # For every test sample...
    for (index, row) in test.iterrows():

        # Report progress.
        if ((len(input_ids) % 1000) == 0):
            print('  Tokenized {:,} comments.'.format(len(input_ids)))

        # Convert sentence pairs to input IDs, with attention masks.
        encoded_dict = tokenizer.encode_plus(row['Report_Content'],  # The text to encode.
                                            max_length=max_len,    # Pad or truncate to this lenght.
                                            pad_to_max_length=True,
                                            truncation=True,
                                            add_special_tokens = True,
                                            return_tensors='pt')   # Return objects as PyTorch tensors.

        # Add this example to our lists.
        input_ids.append(encoded_dict['input_ids'])
        attn_masks.append(encoded_dict['attention_mask'])

    print('\nDONE. {:,} examples.'.format(len(input_ids)))


    # ======== List of Examples --> Tensor ========

    # Convert each Python list of Tensors into a 2D Tensor matrix.
    input_ids = torch.cat(input_ids, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)
    
    labels = test_labels.to_numpy().astype(float)

    # Cast the labels list to a 2D Tensor.
    labels = torch.tensor(labels)

    # ======== Summary ========

    print('\nData structure shapes:')
    print('   input_ids:  {:}'.format(str(input_ids.shape)))
    print('  attn_masks:  {:}'.format(str(attn_masks.shape)))
    print('      labels:  {:}'.format(str(labels.shape)))
    return input_ids, attn_masks, labels


# In[21]:


def eval_model(path, test_dataloader):
    #載入模型
    model_path = path + '/output_epochs20/model_save/'
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_path)
    desc = model.cuda()
    print('Model loaded.')
    #送入模型預測
    predictions, true_labels = model_output(test_dataloader, model)
    #預測結果
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    return flat_predictions, flat_true_labels


# In[22]:


def model_output(test_dataloader, model):
    t0 = time.time()

    # Tracking variables 
    predictions , true_labels = [], []
    Sigmoid_predictions = []
    Sigmoid = nn.Sigmoid()

    print('Evaluating on {:,} test set batches...'.format(len(test_dataloader)))

    # Predict 
    for batch in test_dataloader:

        # Report progress.
        if ((len(predictions) % 500) == 0):
            print('  Batch {:>5,}  of  {:>5,}.'.format(len(predictions), len(test_dataloader)))

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store the compute graph, saving memory 
        # and speeding up prediction
        with torch.no_grad():

            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        Sigmoid_logits = Sigmoid(torch.tensor(logits))
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        Sigmoid_predictions.append(Sigmoid_logits)

    print('    DONE.')

    print('Evaluation took {:.0f} seconds.'.format(time.time() - t0))
    return predictions, true_labels


# In[23]:


def dynamic_thresholds(number, y_pred):
    one_hot = np.zeros_like(y_pred)
    if one_hot.shape == (1, 1):
        return y_pred
    else:
        for i in range(len(y_pred)):
            thresholds = np.max(y_pred[i]) - (number * np.std(y_pred[i]))
            for j in range(len(y_pred[i])):
                if y_pred[i][j] > thresholds:
                    one_hot[i][j] = 1
        return one_hot


# In[25]:


def getDictKeyByValue(dict, value):
    return [k for k, v in dict.items() if v == value]


# In[26]:
folder_path = sys.argv[1]
#讀取傳入json
input_path = os.path.join('./data/', folder_path)
output_path = os.path.join('./output/', folder_path)
with open(input_path, encoding='UTF-8') as f:
    data = json.load(f)

# In[27]:


#取得Finding的句子
content = data['data'][0]['Sentences']


# In[28]:


#預測結果，並轉為json格式
path = './hierarchical ver2'
labels = []
j = {'data' : []}
datas = pd.DataFrame(content, columns=['Report_Content'])
for i in range(len(datas)):
    print(f'{i:*^80}')
    label = []
    data = datas.loc[i : i]
    data = data.reset_index(drop = True)
    Pred(path, data)
    labels.append(label)

for idx, row in datas.iterrows():
    s = {'Sentence' : '', 'Structure' : []}
    print(f'{idx:*^80}')
    print('句子:', row.Report_Content)
    print('預測label:', labels[idx])
    for label in labels[idx]:
        name ='layer'
        cache = []
        nodes = []
        for lab in label:
            try:
                node = getDictKeyByValue(globals()[name], lab)
            except KeyError:
                pass
                #name = 'layer_' + ('_').join(cache[:-1])
                #node = getDictKeyByValue(globals()[name], label)
            cache.append(lab)
            name = 'layer_' + ('_').join(cache)
            nodes.extend(node)
        print('預測結構:', nodes)
        print('完整結構:', (' > ').join(nodes))
        s['Sentence'] = row.Report_Content
        #s['Structure'] = (' > ').join(nodes)
        s['Structure'].append((' > ').join(nodes))
    j['data'].append(s)



with open(output_path, 'w') as f:
    json.dump(j, f)

