#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import torch
def supervit_calculate_difference(accuracy_values):
    layer_combinations = [2, 2*2*2, 2*3*2*3, 2*4*1*3, 2*4*1*3, 3*7*1*3, 4*6*1*1, 3*4*1*1, 2]
    accuracy_matrix = [[] for _ in range(len(layer_combinations))]
    index = 0
    for layer, combinations in enumerate(layer_combinations):
        for _ in range(combinations):
            accuracy_matrix[layer].append(accuracy_values[index])
            index += 1
            
    difference_matrix = [[] for _ in range(len(layer_combinations))]
    for i, layer in enumerate(accuracy_matrix):
        last_element = layer[0]
        difference_layer = [value-last_element for value in layer]
        difference_matrix[i] = difference_layer
 
    return difference_matrix


def supervit_data():
    flops_values=torch.load('./data/supervit/blockflops.pt')
    flops_values=np.array(flops_values[:195])
    flop_difference=supervit_calculate_difference(flops_values)
    accuracy_values=torch.load('./data/supervit/blockacc.pt')
    accuracy_values=np.array(accuracy_values[:195])
    acc_difference=supervit_calculate_difference(accuracy_values)
    return acc_difference, flop_difference


def supertransformer_calculate_difference(encode,decode):
    encodeLoss=encode.reshape((6,6))
    decodeLoss=decode.reshape((6,36))
    decodeLoss1=np.zeros((6,37))
    for i in range(6):
        for j in range(36):
            decodeLoss1[i,j]=decodeLoss[i,j]
    decodeLoss1[1,36]=23.46
    decodeLoss1[2,36]=25.31
    decodeLoss1[3,36]=25.35
    decodeLoss1[4,36]=25.92
    decodeLoss1[5,36]=26.14
    for i in range(6):
        for j in range(6):
            if j==5:
                encodeLoss[i,j]=encodeLoss[i,j]
            if j!=5:    
                encodeLoss[i,j]=encodeLoss[i,5]-encodeLoss[i,j]
    for i in range(6):
        encodeLoss[i,5]=0
    for i in range(6):
        for j in range(37):
            if j==35:
                decodeLoss1[i,j]=decodeLoss1[i,j]
            if j!=35:    
                decodeLoss1[i,j]=decodeLoss1[i,35]-decodeLoss1[i,j]
    for i in range(6):
        decodeLoss1[i,35]=0
        
    encodeLoss_array = np.pad(encodeLoss, ((0, 0), (0, 31)), mode='edge')
    difference=np.concatenate((encodeLoss_array, decodeLoss1), axis=0)
    return difference


def supertransformer_data(device):
    acc_values=torch.load('./data/supertransformer/blockacc.pt')
    acc_values=np.array(acc_values)
    encode=acc_values[:36]
    decode=acc_values[36:-5] 
    acc_difference=supertransformer_calculate_difference(encode,decode)
    filename='./data/supertransformer/'+'blocklat_'+str(device)+'.pt'
    lat_values=torch.load(filename)
    lat_values=np.array(lat_values)
    encode=lat_values[:36]
    decode=lat_values[36:] 
    lat_difference=supertransformer_calculate_difference(encode,decode)
    return acc_difference, lat_difference


def mobilenetv3_data(device):
    if device == 'tx2gpu':
        acc_values=torch.load('./data/mobilenetv3/blockacc_gpu.pt')
    else:
        acc_values=torch.load('./data/mobilenetv3/blockacc_cpu.pt')
    acc_difference=np.array(acc_values)
    filename='./data/mobilenetv3/'+'blocklat_'+str(device)+'.pt'
    lat_values=torch.load(filename)
    lat_difference=np.array(lat_values)
    return acc_difference, lat_difference



def nasbench201_data(device):
    acc_values=torch.load('./data/nasbench201/blockacc.pt')
    acc_difference=np.array(acc_values)
    filename='./data/nasbench201/'+'blocklat_'+str(device)+'.pt'
    lat_values=torch.load(filename)
    lat_difference=np.array(lat_values)
    filename='./data/nasbench201/'+'blockeng_'+str(device)+'.pt'
    eng_values=torch.load(filename)
    eng_difference=np.array(eng_values)
    return acc_difference, lat_difference, eng_difference
