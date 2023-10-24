#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import yaml
import pickle
def supervit_decode(encodearch):
    archdict = {'resolution': 192, 'width': [], 'kernel_size': [], 'expand_ratio': [], 'depth': []}

    # first layer
    w1 = int((encodearch[0] + 2) * 8)
    archdict['width'].append(w1)

    # layer-1
    e2 = encodearch[1]
    w = int(e2 // 4)
    d = int((e2 % 4) // 2)
    k = int(e2 % 2)
    archdict['width'].append((w + 2) * 8)
    archdict['depth'].append(d + 1)
    archdict['kernel_size'].append(k * 2 + 3)
    archdict['expand_ratio'].append(1)

    # layer-2
    e3 = encodearch[2]
    w = int(e3 // 18)
    d = int((e3 % 18) // 6)
    k = int((e3 % 6) // 3)
    e = int(e3 % 3)
    archdict['width'].append((w + 3) * 8)
    archdict['depth'].append(d + 3)
    archdict['kernel_size'].append(k * 2 + 3)
    archdict['expand_ratio'].append(e + 4)

    # layer-3 and layer-4
    for i in range(3, 5):
        ei = encodearch[i]
        w = int(ei // 12)
        d = int((ei % 12) // 3)
        e = int(ei % 3)
        archdict['width'].append((w + 4 + (i - 3) * 4) * 8)
        archdict['depth'].append(d + 3)
        archdict['kernel_size'].append(3)
        archdict['expand_ratio'].append(e + 4)

    # layer-5
    e6 = encodearch[5]
    w = int(e6 // 21)
    d = int((e6 % 21) // 3)
    e = int(e6 % 3)
    archdict['width'].append((w + 14) * 8)
    archdict['depth'].append(d + 3)
    archdict['kernel_size'].append(3)
    archdict['expand_ratio'].append(e + 4)
    # layer-6
    e7 = encodearch[6]
    w = int(e7 // 6)
    d = int(e7 % 6)
    archdict['width'].append((w + 20) * 8)
    archdict['depth'].append(d + 3)
    archdict['kernel_size'].append(3)
    archdict['expand_ratio'].append(6)
    # layer-7
    e8 = encodearch[7]
    w = int(e8 // 4)
    d = int(e8 % 4)
    archdict['width'].append((w + 26) * 8)
    archdict['depth'].append(d + 3)
    archdict['kernel_size'].append(3)
    archdict['expand_ratio'].append(6)
    # last layer
    w = int(encodearch[8])
    archdict['width'].append(w * 192 + 1792)
    return archdict



def supertransformer_decode(archcode):
    decoder_len=len(archcode)-6
    archdict = {'encoder-embed-dim-subtransformer': 512,'decoder-embed-dim-subtransformer': 512,'encoder-ffn-embed-dim-all-subtransformer':[],'decoder-ffn-embed-dim-all-subtransformer':[],'encoder-layer-num-subtransformer':6,'decoder-layer-num-subtransformer':decoder_len,'encoder-self-attention-heads-all-subtransformer':[],'decoder-self-attention-heads-all-subtransformer':[],'decoder-ende-attention-heads-all-subtransformer':[],'decoder-arbitrary-ende-attn-all-subtransformer':[]}
    for j in range(6):
        m = int(archcode[j] // 2)
        n = int(archcode[j] % 2)
        archdict['encoder-ffn-embed-dim-all-subtransformer'].append((m + 1) * 1024)
        archdict['encoder-self-attention-heads-all-subtransformer'].append((n + 1) * 4)
    for j in range(6, len(archcode)):
        if archcode[j] == 36:
            break
        w = int(archcode[j] // 12)
        p = int((archcode[j] % 12) // 6)
        n = int(((archcode[j] % 12) % 6) // 3)
        m = int(((archcode[j] % 12) % 6) % 3)
        archdict['decoder-ffn-embed-dim-all-subtransformer'].append((m + 1) * 1024)
        archdict['decoder-self-attention-heads-all-subtransformer'].append((n + 1) * 4)
        archdict['decoder-ende-attention-heads-all-subtransformer'].append((p + 1) * 4)
        if w == 0:
            archdict['decoder-arbitrary-ende-attn-all-subtransformer'].append(-1)
        else:
            archdict['decoder-arbitrary-ende-attn-all-subtransformer'].append(w)
    return archdict



def supervit_code(archdict):
    encodearch=[]
    #first layer
    e1=int(archdict['width'][0]/8-2)
    encodearch.append(e1)
    #layer-1
    w=int(archdict['width'][1]/8-2)
    d=int(archdict['depth'][0]-1)
    k=int((archdict['kernel_size'][0]-3)/2)
    e2=k+d*2+w*4
    encodearch.append(e2)
    #layer-2
    w=int(archdict['width'][2]/8-3)
    d=int(archdict['depth'][1]-3)
    k=int((archdict['kernel_size'][1]-3)/2)
    e=int(archdict['expand_ratio'][1]-4)
    e3=e+k*3+d*6+w*18
    encodearch.append(e3)
    #layer-3
    w=int(archdict['width'][3]/8-4)
    d=int(archdict['depth'][2]-3)
    #k=int((archdict['kernel_size'][2]-3)/2)
    e=int(archdict['expand_ratio'][2]-4)
    e4=e+d*3+w*12
    encodearch.append(e4)
    #layer-4
    w=int(archdict['width'][4]/8-8)
    d=int(archdict['depth'][3]-3)
    #k=int((archdict['kernel_size'][3]-3)/2)
    e=int(archdict['expand_ratio'][3]-4)
    e5=e+d*3+w*12
    encodearch.append(e5)
    
    #layer-5
    w=int(archdict['width'][5]/8-14)
    d=int(archdict['depth'][4]-3)
    #k=int((archdict['kernel_size'][4]-3)/2)
    e=int(archdict['expand_ratio'][4]-4)
    e6=e+d*3+w*21
    encodearch.append(e6)
    
    #layer-6
    w=int(archdict['width'][6]/8-20)
    d=int(archdict['depth'][5]-3)
    #k=int((archdict['kernel_size'][5]-3)/2)
    #e=int(archdict['expand_ratio'][5]-4)
    e7=d+w*6
    encodearch.append(e7)
    
    #layer-7
    w=int(archdict['width'][7]/8-26)
    d=int(archdict['depth'][6]-3)
    #k=int((archdict['kernel_size'][6]-3)/2)
    #e=int(archdict['expand_ratio'][6]-4)
    e8=d+w*4
    encodearch.append(e8)
    
    #last layer
    w=int((archdict['width'][8]-1792)/192)
    e9=w
    encodearch.append(e9)
    return encodearch


def supertransformer_code(archdict):
    encodearch=[]
    for j in range(6):
        m=archdict['encoder-ffn-embed-dim-all-subtransformer'][j]/1024-1
        n=archdict['encoder-self-attention-heads-all-subtransformer'][j]/4-1
        encodearch.append(m*2+n)
    for j in range(archdict['decoder-layer-num-subtransformer']):
        if (archdict['decoder-arbitrary-ende-attn-all-subtransformer'][j]==-1):
            w=0
        else:
            w=archdict['decoder-arbitrary-ende-attn-all-subtransformer'][j]
        p=archdict['decoder-ende-attention-heads-all-subtransformer'][j]/4-1
        n=archdict['decoder-self-attention-heads-all-subtransformer'][j]/4-1
        m=archdict['decoder-ffn-embed-dim-all-subtransformer'][j]/1024-1
        encodearch.append(w*12+p*6+n*3+m)
    if len(encodearch)!=12:
        encodearch.append(36)
    return encodearch




def mobilenetv3_decode(encodearch, device):
    blocklist=[]
    if device == 'tx2gpu':
        filename='./data/mobilenetv3/blocksGPU/'
    else:
        filename='./data/mobilenetv3/blocksCPU/'
    for i in range(5):
        block_filename=filename+'block'+str(i+1)+'.pickle'
        file=open(block_filename,"rb")
        b=pickle.load(file)
        blocklist.append(b)
        
    ks=[]
    d=[]
    e=[]
    for i in range(5):
        m=encodearch[i]
        Block=blocklist[i]['config'][m]
        ks+=Block['ks']
        d.append(Block['d'])
        e+=Block['e']
    
    archdict={}
    archdict['ks']=ks
    archdict['d']=d
    archdict['e']=e
    archdict['encodearch']=encodearch
    return archdict



def nasbench201_decode_help(code_number):
    if code_number == 0:
        return 'none'
    if code_number == 1:
        return 'skip_connect'
    if code_number == 2:
        return 'nor_conv_1x1'
    if code_number == 3:
        return 'nor_conv_3x3'
    if code_number == 4:
        return 'avg_pool_3x3'
    
    
def nasbench201_decode(encodearch):
    archdict={}
    codestr='|'+nasbench201_decode_help(encodearch[0])+'~0|+|'+nasbench201_decode_help(encodearch[1])+'~0|'+nasbench201_decode_help(encodearch[2])+'~1|+|'+nasbench201_decode_help(encodearch[3])+'~0|'+nasbench201_decode_help(encodearch[4])+'~1|'+nasbench201_decode_help(encodearch[5])+'~2|'
    archdict['archstr']=codestr
    archdict['archcode']=encodearch
    return archdict