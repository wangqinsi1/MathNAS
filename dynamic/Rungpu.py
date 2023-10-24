import importlib
import os
import os.path as osp
import argparse
import numpy as np
import math 
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets
import gc
from ofa.utils import AverageMeter, accuracy
from ofa.model_zoo import ofa_net
from ofa.model_zoo import ofa_specialized
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
import pickle
#import psutil
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
import numpy as np
import time
#from scipy.optimize import linprog
import numpy as np
from math import floor, ceil
import copy
import random 

    
def blockchange(ofa_net,net,net_new,net_old,changenum):
    old_d=net_old['d']
    new_d=net_new['d']
    start=0
    for i in range(changenum):
        start+=new_d[i]
    d1=net_new['d']
    for i in range(5):
        if i!=changenum:
            d1[i]=0
    if (changenum==1)or(changenum==2)or(changenum==3):
        d1[changenum-1]=1
        d1[changenum+1]=1
    if (changenum==4):
        d1[changenum-1]=1
    if (changenum==0):
        d1[changenum+1]=1
    knew=[]
    enew=[]
    for i in range(5):
        for j in range(4):
            if (i==changenum):
                m=net_new['ks'][i*4+j]
                knew.append(m)
                n=net_new['e'][i*4+j]
                enew.append(n)
            else:
                knew.append(3)
                enew.append(3)
    ofa_net.set_active_subnet(ks=knew, d=d1, e=enew)
    subblock1 = ofa_net.get_active_subnet()
    blockuse=[]
    for i in range(new_d[changenum]):
        if changenum!=0:
            blockuse.append(subblock1.blocks[1+i+1])
        if changenum==0:
            blockuse.append(subblock1.blocks[i+1])
    subblock=blockuse
    net.blocks[start+1]=subblock[0]
    net.blocks[start+2]=subblock[1]
    if old_d[changenum]==new_d[changenum]:
        if(new_d[changenum]==3):
            net.blocks[start+3]=subblock[2]
        if(new_d[changenum]==4):
            net.blocks[start+3]=subblock[2]
            net.blocks[start+4]=subblock[3]
    if old_d[changenum]>new_d[changenum]:
        if(new_d[changenum]==2)and (old_d[changenum]==3):
            net.blocks[start+3] = nn.Sequential()
        if(new_d[changenum]==2)and (old_d[changenum]==4):
            net.blocks[start+3] = nn.Sequential()   
            net.blocks[start+4] = nn.Sequential()  
        if(new_d[changenum]==3)and (old_d[changenum]==4):
            net.blocks[start+3] = subblock[2]  
            net.blocks[start+4] = nn.Sequential() 
    if old_d[changenum]<new_d[changenum]:
        if(new_d[changenum]==3)and (old_d[changenum]==2):
            net.blocks.insert(start+3,subblock[2])
        if(new_d[changenum]==4)and (old_d[changenum]==2):
            net.blocks.insert(start+3,subblock[2])
            net.blocks.insert(start+4,subblock[3])
        if(new_d[changenum]==4)and (old_d[changenum]==3):
            net.blocks[start+3] = subblock[2] 
            net.blocks.insert(start+4,subblock[3])    
    return net
    
def change(ofa_net,net_new):
    ofa_net.set_active_subnet(ks=net_new['ks'], d=net_new['d'], e=net_new['e'])
    net = ofa_net.get_active_subnet()
    return net 

def makeconfig(blockchoose):
    file=open(r"/GPUblock/block1.pickle","rb")
    b1=pickle.load(file)
    file=open(r"/GPUblock/block2.pickle","rb")
    b2=pickle.load(file)
    file=open(r"/GPUblock/block3.pickle","rb")
    b3=pickle.load(file)
    file=open(r"/GPUblock/block4.pickle","rb")
    b4=pickle.load(file)
    file=open(r"/GPUblock/block5.pickle","rb")
    b5=pickle.load(file)
    blocklist=[b1,b2,b3,b4,b5]
    ks=[]
    d=[]
    e=[]
    for i in range(5):
        m=blockchoose[i]
        Block=blocklist[i]['config'][m]
        ks+=Block['ks']
        d.append(Block['d'])
        e+=Block['e']
    
    netconfig={}
    netconfig['ks']=ks
    netconfig['d']=d
    netconfig['e']=e
    return netconfig

def latencyupdate(netconfig,lat):
    latency= np.loadtxt('latencyGPU.txt', dtype=np.float32, delimiter=' ')  
    latencybase=18.3
    latencypredict=latencybase
    for i in range(5):
        latencypredict+=latency[i,netconfig[i]]
    m=lat/latencypredict
    latencynew=latency*m
    latbasenew=latencybase*m
    print('m=',m)
    return latencynew,latbasenew

def validateinit(net, path, image_size, data_loader,batch_size=16):
    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    net.eval() 
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    latency = []
    #print('Start inference')
    record = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images, labels
            st = time.time()
            output = net(images)
            ed = time.time()
            print('this epoch latency is ',(ed-st)*1000)

            if (i>=5):
                latency.append((ed-st)*1000)
            if (i==50):
                break 
            output = output.view(-1,1000)
                
           #print(output.size(),labels.size())
            loss = criterion(output, labels)
                # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
 
    return np.mean(latency)




def validate(net, path, image_size, data_loader,latencylimit,latencymin,batch_size=100):
    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    net.eval() 
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    latency = []
    #print('Start inference')
    record = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images, labels
            st = time.time()
            output = net(images)
            ed = time.time()
            
            print('this epoch latency is ',(ed-st)*1000)
            if (i>=10):
                latency.append((ed-st)*1000)
            if (i%20==0) and (i!=0):
                print(i,np.mean(latency))
                if(np.mean(latency)>latencylimit):
                    print('changenet')
                    return np.mean(latency)
                if(np.mean(latency)<latencymin):
                    return np.mean(latency)
                latency=[] 
            output = output.view(-1,1000)
                
                #print(output.size(),labels.size())
            loss = criterion(output, labels)
                # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
 
    print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
    return np.mean(latency)

def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
def Run(latencylimit,latencymin=0,ILP):
    if ILP==0:
        from Linprog import search
    if ILP==1:
        from Gurobipy import search
    imagenet_data_path = '/dataset/test/'
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root= imagenet_data_path,
            transform=build_val_transform(224)
            ),
        batch_size=16,  # test batch size
        shuffle=False,
        num_workers=62,  # number of workers for the data loader
        pin_memory=False,
        drop_last=False,
        )
    data_loader.dataset
    torch.set_num_threads(1)
    #usage1 = psutil.virtual_memory().percent
    latency= np.loadtxt('latencyGPU.txt', dtype=np.float32, delimiter=' ') 
    config=makeconfig([0,0,0,0,0])
    ofa_net1= ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
    net=change(ofa_net1,config)
    latnow = validateinit(net,imagenet_data_path,224,data_loader,batch_size=16)
    print('lat now is',latnow)
    latency_update,latbase_update = latencyupdate([0,0,0,0,0],latnow)
    T1=time.time()
    arch,netchange,changenum=search(latencylimit,10000,latency_update,latbase_update,[0,0,0,0,0])
    T2=time.time()
    print('Init search time is ',(T2-T1)*1000,'ms')
    print('init arch is',arch)
    config=makeconfig(arch)
    T3=time.time()
    net=change(ofa_net1,config)
    T4=time.time()
    print('Init upload time is ',(T4-T3)*1000,'ms')
    print('start inference')
    while True:
        latnow = validate(net,imagenet_data_path,224,data_loader,latencylimit,latencymin,batch_size=16)
        print('latency is',latnow)
        latency_update,latbase_update = latencyupdate(arch,latnow)
        t1=time.time()
        arch,netchange,changenum = search(latencylimit,latnow,latency_update,latbase_update,arch)
        t2=time.time()
        print('search time is',(t2-t1)*1000,'ms')
        print('now arch is',arch)
        configold=config
        config=makeconfig(arch)
        t3=time.time()
        if netchange==1:
            net=change(ofa_net1,config)
        if netchange==0:
            net=blockchange(ofa_net1,net,config,configold,changenum)
        t4=time.time()
        print('upload net time is',(t4-t3)*1000,'ms')

