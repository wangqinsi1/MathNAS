#!/usr/bin/env python
# coding: utf-8

# In[9]:


import yaml
from utils import supervit_code, supertransformer_code
def supervit_predict(acc_difference, flop_difference, arch_path):
    with open(arch_path, 'r') as file:
        archdict = yaml.load(file, Loader=yaml.FullLoader)
    print('The net arch is',archdict) 
    archcode = supervit_code(archdict)
    acc_predict=82.6-sum(acc_difference[i][int(archcode[i])] for i in range(len(acc_difference))) 
    flop_predict=407-sum(flop_difference[i][int(archcode[i])] for i in range(len(flop_difference))) 
    print('The predicted accuracy of the net is {:.1f} %'.format(acc_predict))
    print('The predicted latency of the net is {:.1f} M'.format(flop_predict))
    return acc_predict, flop_predict


def supertransformer_predict(acc_difference, lat_difference, device, arch_path):
    with open(arch_path, 'r') as file:
        archdict = yaml.load(file, Loader=yaml.FullLoader)
    print('The net arch is',archdict) 
    archcode = supertransformer_code(archdict)
    acc_predict = 26.61-sum(acc_difference[i][int(archcode[i])] for i in range(len(acc_difference)))
    lat_base = 9200 if device == 'raspberry pi' else 407.2 if device == 'xeon' else 274.4
    lat_predict=lat_base-sum(lat_difference[i][int(archcode[i])] for i in range(len(lat_difference))) 
    print('The predicted accuracy of the net is {:.1f} %'.format(acc_predict))
    print('The predicted latency of the net is {:.1f} s'.format(lat_predict))
    return acc_predict, lat_predict

def mobilenetv3_predict(acc_difference, lat_difference, device, arch_path):
    with open(arch_path, 'r') as file:
        archdict = yaml.load(file, Loader=yaml.FullLoader)
    print('The net arch is',archdict) 
    archcode = archdict['encodearch']
    acc_predict = 79.23-sum(acc_difference[i][int(archcode[i])] for i in range(len(acc_difference)))
    lat_predict = sum(lat_difference[i,archcode[i]] for i in range(5))
    print('The predicted accuracy of the net is {:.1f} %'.format(acc_predict))
    print('The predicted latency of the net is {:.1f} s'.format(lat_predict))
    return acc_predict, lat_predict

def nasbench201_predict(acc_difference, lat_difference, eng_difference, device, arch_path):
    with open(arch_path, 'r') as file:
        archdict = yaml.load(file, Loader=yaml.FullLoader)
    print('The net arch is',archdict['archstr']) 
    archcode = archdict['archcode']
    acc_predict = 40.78-sum(acc_difference[i][int(archcode[i])] for i in range(len(acc_difference)))
    lat_base = 5711 if device == 'edgegpu' else 899
    eng_base = 23834 if device == 'edgegpu' else 6291
    lat_predict = lat_base-sum(lat_difference[i,archcode[i]] for i in range(6))
    eng_predict = eng_base-sum(eng_difference[i,archcode[i]] for i in range(6))
    print('The predicted accuracy of the net is {:.1f} %'.format(acc_predict))
    print('The predicted latency of the net is {:.1f} s'.format(lat_predict))
    print('The predicted energy of the net is {:.1f} s'.format(eng_predict))
    return acc_predict, lat_predict, eng_predict

