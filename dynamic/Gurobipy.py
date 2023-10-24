import os
import os.path as osp
import argparse
import numpy as np
import math 
import time  
import gc 
import pickle
import psutil 
import numpy as np
import time
from scipy.optimize import linprog
import numpy as np
from math import floor, ceil
import copy
import random
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB

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
    
    if topkmax==0:
        netchange=1
        print('start heavy search')
        model = gp.Model("mip1")
        x = model.addVars(25, name = 'x', vtype = GRB.BINARY)
        topk1all= np.loadtxt('topk1all.txt', dtype=np.float32, delimiter=' ') 
        topk1=topk1all
        latencylimit1=latencylimit-latencybase
        model.setObjective(79.23-(x[0]*topk1[0,0]+x[1]*topk1[0,1]+x[2]*topk1[0,2]+x[3]*topk1[0,3]+x[4]*topk1[0,4]+x[5]*topk1[1,0]+x[6]*topk1[1,1]+x[7]*topk1[1,2]+x[8]*topk1[1,3]+x[9]*topk1[1,4]+x[10]*topk1[2,0]+x[11]*topk1[2,1]+x[12]*topk1[2,2]+x[13]*topk1[2,3]+x[14]*topk1[2,4]+x[15]*topk1[3,0]+x[16]*topk1[3,1]+x[17]*topk1[3,2]+x[18]*topk1[3,3]+x[19]*topk1[3,4]+x[20]*topk1[4,0]+x[21]*topk1[4,1]+x[22]*topk1[4,2]+x[23]*topk1[4,3]+x[24]*topk1[4,4]), GRB.MAXIMIZE)
        model.addConstr(x[0] + x[1] + x[2] + x[3] + x[4] == 1, "c0")
        model.addConstr(x[5] + x[6] + x[7] + x[8] + x[9] == 1, "c1")
        model.addConstr(x[10] + x[11] + x[12] + x[13] + x[14] == 1, "c2")
        model.addConstr(x[15] + x[16] + x[17] + x[18] + x[19] == 1, "c3")
        model.addConstr(x[20] + x[21] + x[22] + x[23] + x[24] == 1, "c4")
        model.addConstr(x[0]*latency[0,0]+x[1]*latency[0,1]+x[2]*latency[0,2]+x[3]*latency[0,3]+x[4]*latency[0,4]+x[5]*latency[1,0]+x[6]*latency[1,1]+x[7]*latency[1,2]+x[8]*latency[1,3]+x[9]*latency[1,4]+x[10]*latency[2,0]+x[11]*latency[2,1]+x[12]*latency[2,2]+x[13]*latency[2,3]+x[14]*latency[2,4]+x[15]*latency[3,0]+x[16]*latency[3,1]+x[17]*latency[3,2]+x[18]*latency[3,3]+x[19]*latency[3,4]+x[20]*latency[4,0]+x[21]*latency[4,1]+x[22]*latency[4,2]+x[23]*latency[4,3]+x[24]*latency[4,4]<= latencylimit1, "c5")
        model.optimize()
        model.write("model_integer.lp")

        upper_bound, lower_bound = float('inf'), 0
        model_relax = model.relax()
        root_node = Node(model = model_relax, upper_bound = upper_bound, lower_bound = lower_bound, candidate_vars = [i for i in range(model.NumVars)])
        candidate_node = [root_node]
        current_optimum = None

        while candidate_node:
            node, candidate_node = choice_node(candidate_node)
            if node.upper_bound <= lower_bound:
                #print("prune by bound")
                continue
            model_status = node.optimize(heuristic_solve)
            if model_status == 'infeasible':
                #print("prune by infeasiblity")
                continue
            node.update_upper_bound()
            if node.upper_bound <= lower_bound:
                #print("prune by bound")
                continue
            if node.is_integer():
                node.update_lower_bound()
                if node.lower_bound > lower_bound:
                    lower_bound = node.lower_bound
                    current_optimum = node.solution
                continue
            if node.is_child_problem():
                child_node1, child_node2 = node.get_child_problem()
                candidate_node.append(child_node1)
                candidate_node.append(child_node2)
        
        print('the new acc is',lower_bound)
        archlast=[]
        for v in model.getVars():
            archlast.append(v.x)
        archnow=[]
        for i in range(5):
            dex=archlast[i*5:i*5+5]
            archnow.append(dex.index(1))
        latencynew=0
        for i in range(5):
            latencynew=latencynew+latency[i,archnow[i]]
        latencynew=latencynew+latencybase    
    print('the new latency is',latencynew)
    T2 = time.time()   
    return archnow,netchange,changenum

