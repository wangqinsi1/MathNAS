import importlib
import os
import os.path as osp
import argparse
import numpy as np
import math 
import time 
from torchvision import transforms, datasets
import gc 
import pickle  
import numpy as np 
from scipy.optimize import linprog
import numpy as np
from math import floor, ceil
import copy
import random

def search(latencylimit,latencynow,latency,latbase,basenet):
    topk1all= np.loadtxt('topk1all.txt', dtype=np.float32, delimiter=' ') 
    basetopk1=79.23
    archloss=np.array([[21,27,36,48,60],[48,60,81,95,156],[39,60,76,108,154],[30,53,64,88,144],[27,36,50,66,106]])
    topkmax=0
    latencynew=0
    T1 = time.time()
    netchange=0
    changenum=0
    latencybase=latbase
    #light search
    if latencylimit<latencynow:
        for i in range(5):
            if basenet[i]!=0:
                lat=latencynow-latency[i,basenet[i]]+latency[i,basenet[i]-1]
                if lat<=latencylimit:
                    arch=copy.deepcopy(basenet)
                    arch[i]=basenet[i]-1
                    topk1=basetopk1-(topk1all[0,int(arch[0])]+topk1all[1,int(arch[1])]+topk1all[2,int(arch[2])]+topk1all[3,int(arch[3])]+topk1all[4,int(arch[4])])/(0.19*((archloss[0,int(arch[0])]+archloss[1,int(arch[1])]+archloss[2,int(arch[2])]+archloss[3,int(arch[3])]+archloss[4,int(arch[4])])**0.3))
                    if topk1>topkmax:
                        changenum=i
                        topkmax=topk1
                        latencynew=lat
                        archnow=arch
    if latencylimit>latencynow:
        for i in range(5):
            if basenet[i]!=4:
                lat=latencynow-latency[i,basenet[i]]+latency[i,basenet[i]+1]
                if lat<=latencylimit:
                    arch=copy.deepcopy(basenet)
                    arch[i]=basenet[i]+1
                    topk1=basetopk1-(topk1all[0,int(arch[0])]+topk1all[1,int(arch[1])]+topk1all[2,int(arch[2])]+topk1all[3,int(arch[3])]+topk1all[4,int(arch[4])])/(0.19*((archloss[0,int(arch[0])]+archloss[1,int(arch[1])]+archloss[2,int(arch[2])]+archloss[3,int(arch[3])]+archloss[4,int(arch[4])])**0.3))
                    if topk1>topkmax:
                        changenum=i
                        topkmax=topk1
                        latencynew=lat
                        archnow=arch
    
    #heavy search
    if topkmax==0:
        netchange=1
        print('start heavy search')
        topk1=topk1all
        integer_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        c = [topk1[0,0],topk1[0,1],topk1[0,2],topk1[0,3],topk1[0,4],topk1[1,0],topk1[1,1],topk1[1,2],topk1[1,3],topk1[1,4],topk1[2,0],topk1[2,1],topk1[2,2],topk1[2,3],topk1[2,4],topk1[3,0],topk1[3,1],topk1[3,2],topk1[3,3],topk1[3,4],topk1[4,0],topk1[4,1],topk1[4,2],topk1[4,3],topk1[4,4]]
        A = [[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],[-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1],[latency[0,0],latency[0,1],latency[0,2],latency[0,3],latency[0,4],latency[1,0],latency[1,1],latency[1,2],latency[1,3],latency[1,4],latency[2,0],latency[2,1],latency[2,2],latency[2,3],latency[2,4],latency[3,0],latency[3,1],latency[3,2],latency[3,3],latency[3,4],latency[4,0],latency[4,1],latency[4,2],latency[4,3],latency[4,4]]]
        b = [1,1,1,1,1,-0.5,-0.5,-0.5,-0.5,-0.5,latencylimit-latencybase]
        x_bounds = [[0, 1] for _ in range(len(c))]
        bb_algorithm = BbAlgorithm(c, A, b, x_bounds, integer_val)
        x=bb_algorithm.run()
        archnow=[]
        for i in range(5):
            dex=x[i*5:i*5+5].tolist()
            archnow.append(dex.index(1))
        latencynew=0
        for i in range(5):
            latencynew=latencynew+latency[i,archnow[i]]
        latencynew+=latencybase
            
    print('the new latency is',latencynew)
    
    T2 = time.time()   
    return archnow,netchange,changenum










class Node(object):
    def __init__(self, x_bounds=[], freeze_var_list=[], index=0, upper_or_lower=0):
        self._x_bounds = x_bounds
        self._freeze_var_list = freeze_var_list
        self._index = index
        self._upper_or_lower = upper_or_lower
 
 
    def freeze_var(self, index, val):
        self._x_bounds[index][0] = val
        self._x_bounds[index][1] = val
        self._freeze_var_list.append(index)
 
    def set_lp_res(self, res):
        self._res = res
        s = ""
        for l in range(len(self._res['x'])):
            if l in self._freeze_var_list:
                s += "[" + str(self._res['x'][l]) + "]"
            else:
                s += " " + str(self._res['x'][l])
         
 
    def check_integer_val_solved(self, m):
        return True if m == len(self._freeze_var_list) else False
 
 
class BbAlgorithm(object):
    def __init__(self, c, a_ub, b_ub, x_b, integer_val):
        self.c = c
        self.a_ub = a_ub
        self.b_ub = b_ub
        self.x_b = x_b
        self._integer_val = integer_val
        self.best_solution = float('inf')
        self.best_node = None
        self.nodes = []
        self.nodes_solution = []
 
    def solve_lp(self, cur_x_b):
        return linprog(self.c, A_ub=self.a_ub, b_ub=self.b_ub, bounds=cur_x_b)
 
    def check_fessible(self, res):
        if res['status'] == 0:
            return True
        elif res['status'] == 2:
            return False
        else:
            raise ("Error")
 
    def add_node(self, node):
        res = self.solve_lp(node._x_bounds)
        if self.check_fessible(res) and res['fun'] < self.best_solution:
            node.set_lp_res(res)
            self.nodes_solution.append(res['fun'])
            self.nodes.append(node)
            if node.check_integer_val_solved(len(self._integer_val)):
                self.best_solution = res['fun']
                self.best_node = node        
            return True
        else:
            return False
 
    def del_higher_val_node(self, z_s):
        del_list = []
        for i in range(len(self.nodes_solution)):
            if self.nodes_solution[i] >= z_s:
                del_list.append(i)
        s = ""
        for i in del_list:
            s += " " + str(self.nodes[i]._index)
        
        self.nodes = list(np.delete(self.nodes, del_list))
        self.nodes_solution = list(np.delete(self.nodes_solution, del_list))
        
 
    def del_item(self, index):
         
        self.nodes = list(np.delete(self.nodes, index))
        self.nodes_solution = list(np.delete(self.nodes_solution, index))
        
 
    def check_bounds(self, temp_x_b, index, u_or_l):
        if u_or_l == 1:
            if self.x_b[index][0] is not None and temp_x_b[index][0] is None:
                return False
            elif self.x_b[index][0] is None and temp_x_b[index][0] is not None:
                return True
            elif self.x_b[index][0] is not None and temp_x_b[index][0] is not None:
                return False if(self.x_b[index][0] > temp_x_b[index][0]) else True
        elif u_or_l == 2:
            if self.x_b[index][1] is not None and temp_x_b[index][1] is None:
                return False
            elif self.x_b[index][1] is None and temp_x_b[index][1] is not None:
                return True
            elif self.x_b[index][1] is not None and temp_x_b[index][1] is not None:
                return False if(self.x_b[index][1] < temp_x_b[index][1]) else True
        else:
             
            exit()

