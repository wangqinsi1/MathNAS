#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import math
import pickle
import argparse
import yaml
from NASViT.main import start, generate_model

parser = argparse.ArgumentParser() 
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--imagenet_save_path', type=str, default=None)

args = parser.parse_args()

with open(args.config_path, 'r') as file:
    subconfig = yaml.load(file, Loader=yaml.FullLoader)
    
config, model, data_loader_train, data_loader_val = start(args.imagenet_save_path)
           
acc1, acc5, loss, flops, params= generate_model(subconfig, config, model, data_loader_train, data_loader_val)

print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, acc1, acc5))

