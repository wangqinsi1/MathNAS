#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from utils import supervit_decode, supertransformer_decode, mobilenetv3_decode, nasbench201_decode

def heuristic_solve(problem):
    problem.Params.OutputFlag = 0
    problem.optimize()
    if problem.status == GRB.INFEASIBLE:
        return None, None
    return problem.ObjVal, problem.getVars()

def choice_node(condidate_node):
    node = condidate_node.pop(0)
    return node, condidate_node

class Node:
    def __init__(self, model, upper_bound, lower_bound, candidate_vars):
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        self.model = model
        self.candidate_vars = candidate_vars.copy()
        assert(upper_bound >= lower_bound), "upper bound is less than lower bound"

    def optimize(self, heuristic_solve):
        self.obj_values, self.solution = heuristic_solve(self.model)
        if self.obj_values == None:
            return "infeasible"
        return "feasible"
    
    def update_upper_bound(self):
        if self.upper_bound > self.obj_values:
            self.upper_bound = self.obj_values
            assert(self.lower_bound <= self.obj_values)
            assert(self.lower_bound <= self.upper_bound), "upper bound is less than lower bound"
    
    def update_lower_bound(self):
        self.lower_bound = self.obj_values
        assert(self.lower_bound <= self.obj_values)
        assert(self.lower_bound <= self.upper_bound), "upper bound is less than lower bound"
    
    def is_integer(self):
        for var in self.solution:
            if 0 < var.x and var.x < 1:
                return False
        return True
    
    def is_child_problem(self):
        if self.candidate_vars:
            return True
    
    def get_child_problem(self):
        self.child_left, self.child_right = self.model.copy(), self.model.copy()
        branch_index, self.condidate_child_vars = self.choice_branch(self.candidate_vars)
        self.child_left.addConstr(self.child_left.getVars()[branch_index] == 0)
        self.child_right.addConstr(self.child_right.getVars()[branch_index] == 1)
        node_left = Node(self.child_left, self.upper_bound, self.lower_bound, self.condidate_child_vars)
        node_right = Node(self.child_right, self.upper_bound, self.lower_bound, self.condidate_child_vars)
        return node_left, node_right
    
    def choice_branch(self, candidate_vars):
        self.condidate_child_vars = self.candidate_vars.copy()
        branch_index = self.condidate_child_vars.pop(0)
        return branch_index, self.condidate_child_vars
    
    def write(self):
        self.model.write("model.lp")


# In[10]:


def search(model):
    print('==============================START SEARCH============================')
    T1 = time.time() 
    model.Params.TimeLimit = 5   
    model.optimize()
    model.write("model_integer.lp")

    upper_bound, lower_bound = float('inf'), 0
    model_relax = model.relax()
    root_node = Node(model = model_relax, upper_bound = upper_bound, lower_bound = lower_bound, candidate_vars = [i for i in range(model.NumVars)])
    candidate_node = [root_node]
    current_optimum = None
    start_time = time.time()
    time_limit=10

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
        
        if model.Status == GRB.Status.TIME_LIMIT:
            print("Time limit reached. The best solution found is:")
            for v in model.getVars():
                print(f"{v.VarName}: {v.X}")
            print(f"Objective value: {model.ObjVal}")
            break  
            
        elif time.time() - start_time >= time_limit:
            print("Time limit reached. Stopping the loop.")
            break
        else:
            pass
    T2 = time.time() 
    print('==============================END SEARCH============================')
    print('search time is {:.5f} s'.format(T2-T1)) 
    return model,model.ObjVal


# In[19]:


def supervit_search(acc_difference,flop_difference,floplimit):
    #model building
    topk1= [item for sublist in acc_difference for item in sublist]
    flop= [item for sublist in flop_difference for item in sublist]
    model = gp.Model("mip1")
    x = model.addVars(195, name = 'x', vtype = GRB.BINARY)
    expr =82.6 -sum(x[i] * topk1[i] for i in range(195)) 
    model.setObjective(expr, GRB.MAXIMIZE)
    model.addConstr(sum(x[i] for i in range(2)) == 1, "c0")
    model.addConstr(sum(x[i+2] for i in range(8)) == 1, "c1")
    model.addConstr(sum(x[i+10] for i in range(36)) == 1, "c2")
    model.addConstr(sum(x[i+46] for i in range(24)) == 1, "c3")
    model.addConstr(sum(x[i+70] for i in range(24)) == 1, "c4")
    model.addConstr(sum(x[i+94] for i in range(63)) == 1, "c5")
    model.addConstr(sum(x[i+157] for i in range(24)) == 1, "c6")
    model.addConstr(sum(x[i+181] for i in range(12)) == 1, "c7")
    model.addConstr(sum(x[i+193] for i in range(2)) == 1, "c8")
    model.addConstr(sum(x[i] * flop[i] for i in range(195)) <= floplimit, "c9")
    #search
    model,accnew=search(model)
    #encode
    archnew=[]
    for v in model.getVars():
        archnew.append(v.x)
    flopnew=sum(archnew[i] * flop[i] for i in range(195)) 
    archcode=[]
    elements_per_row = [2,8,36,24,24,63,24,12,2]
    index = 0
    for count in elements_per_row:
        archcode.append(archnew[index:index + count])
        index += count
    ARCH=[]
    for i in range(len(archcode)):
        m= np.argmax(archcode[i])
        ARCH.append(m)
    ARCHcode=supervit_decode(ARCH)
    print('The search net is',ARCHcode)
    print('The flop of search net is {:.1f} M'.format(flopnew))
    #save net
    return ARCHcode, accnew, flopnew


# In[18]:


def supertransformer_search(topk1,latency,latencylimit,device):
    print('start heavy search')
    model = gp.Model("mip1")
    x = model.addVars(258, name = 'x', vtype = GRB.BINARY)
    latencylimit1=latencylimit
    m, n = topk1.shape
    lat_base = 9200 if device == 'raspberry pi' else 407.2 if device == 'xeon' else 274.4
    expr = 26.61 -sum(x[i * 6 + j] * topk1[i, j] for i in range(6) for j in range(6))-sum(x[i * 37 + j + 36] * topk1[i+6, j] for i in range(6) for j in range(37))
    
    model.setObjective(expr, GRB.MAXIMIZE)
    model.addConstr(sum(x[i] for i in range(6)) == 1, "c0")
    model.addConstr(sum(x[i+6] for i in range(6)) == 1, "c1")
    model.addConstr(sum(x[i+12] for i in range(6)) == 1, "c2")
    model.addConstr(sum(x[i+18] for i in range(6)) == 1, "c3")
    model.addConstr(sum(x[i+24] for i in range(6)) == 1, "c4")
    model.addConstr(sum(x[i+30] for i in range(6)) == 1, "c5")
    model.addConstr(sum(x[i+36] for i in range(36)) == 1, "c6")
    model.addConstr(x[72]+sum(x[i+73] for i in range(37)) == 1, "c7")
    model.addConstr(x[72]+x[109]+sum(x[i+110] for i in range(37)) == 1, "c8")
    model.addConstr(x[72]+x[109]+x[146]+sum(x[i+147] for i in range(37)) == 1, "c9")
    model.addConstr(x[72]+x[109]+x[146]+x[183]+sum(x[i+184] for i in range(37)) == 1, "c10")
    model.addConstr(x[72]+x[109]+x[146]+x[183]+x[220]+sum(x[i+221] for i in range(37)) == 1, "c11")
    model.addConstr(lat_base -1.0*sum(x[i * 6 + j] * latency[i, j] for i in range(6) for j in range(6))-1.2*sum(x[i * 37 + j + 36] * latency[i+6, j] for i in range(6) for j in range(37))<= latencylimit, "c12")
    model,accnew=search(model)
    
    archlast=[]
    for v in model.getVars():
        archlast.append(v.x)
    archnow=[]
    for i in range(6):
        dex=archlast[i*6:i*6+6]
        archnow.append(dex.index(1))
    for i in range(6):
        dex=archlast[36+i*37:36+i*37+37]
        archnow.append(dex.index(1))
        if archlast[36+i*37+36]==1:
            break

    latencynew=lat_base -1.0*sum(archlast[i * 6 + j] * latency[i, j] for i in range(6) for j in range(6))-1.2*sum(archlast[i * 37 + j + 36] * latency[i+6, j] for i in range(6) for j in range(37))  
    ARCHcode=supertransformer_decode(archnow)
    print('The search net is',ARCHcode) 
    print('The latency of search net is {:.1f} s'.format(latencynew))
    return ARCHcode, accnew, latencynew




def mobilenetv3_search(topk1,latency,latencylimit,device):
    print('start heavy search')
    model = gp.Model("mip1")
    x = model.addVars(25, name = 'x', vtype = GRB.BINARY)
    model.setObjective(79.23-(x[0]*topk1[0,0]+x[1]*topk1[0,1]+x[2]*topk1[0,2]+x[3]*topk1[0,3]+x[4]*topk1[0,4]+x[5]*topk1[1,0]+x[6]*topk1[1,1]+x[7]*topk1[1,2]+x[8]*topk1[1,3]+x[9]*topk1[1,4]+x[10]*topk1[2,0]+x[11]*topk1[2,1]+x[12]*topk1[2,2]+x[13]*topk1[2,3]+x[14]*topk1[2,4]+x[15]*topk1[3,0]+x[16]*topk1[3,1]+x[17]*topk1[3,2]+x[18]*topk1[3,3]+x[19]*topk1[3,4]+x[20]*topk1[4,0]+x[21]*topk1[4,1]+x[22]*topk1[4,2]+x[23]*topk1[4,3]+x[24]*topk1[4,4]), GRB.MAXIMIZE)
    model.addConstr(x[0] + x[1] + x[2] + x[3] + x[4] == 1, "c0")
    model.addConstr(x[5] + x[6] + x[7] + x[8] + x[9] == 1, "c1")
    model.addConstr(x[10] + x[11] + x[12] + x[13] + x[14] == 1, "c2")
    model.addConstr(x[15] + x[16] + x[17] + x[18] + x[19] == 1, "c3")
    model.addConstr(x[20] + x[21] + x[22] + x[23] + x[24] == 1, "c4")
    model.addConstr(x[0]*latency[0,0]+x[1]*latency[0,1]+x[2]*latency[0,2]+x[3]*latency[0,3]+x[4]*latency[0,4]+x[5]*latency[1,0]+x[6]*latency[1,1]+x[7]*latency[1,2]+x[8]*latency[1,3]+x[9]*latency[1,4]+x[10]*latency[2,0]+x[11]*latency[2,1]+x[12]*latency[2,2]+x[13]*latency[2,3]+x[14]*latency[2,4]+x[15]*latency[3,0]+x[16]*latency[3,1]+x[17]*latency[3,2]+x[18]*latency[3,3]+x[19]*latency[3,4]+x[20]*latency[4,0]+x[21]*latency[4,1]+x[22]*latency[4,2]+x[23]*latency[4,3]+x[24]*latency[4,4]<= latencylimit, "c5")
    model,accnew=search(model)
    
    archlast=[]
    for v in model.getVars():
        archlast.append(v.x)
    archnow=[]
    for i in range(5):
        dex=archlast[i*5:i*5+5]
        archnow.append(dex.index(1))
    
    ARCHcode=mobilenetv3_decode(archnow, device)
    print('The search net is',ARCHcode) 
    latencynew = sum(latency[i,archnow[i]] for i in range(5))
    print('The latency of search net is {:.1f} s'.format(latencynew))
    return ARCHcode, accnew, latencynew



def nasbench201_search(topk1,latency,energy, latencylimit,energylimit,device):
    lat_base = 5711 if device == 'edgegpu' else 899
    eng_base = 23834 if device == 'edgegpu' else 6291
    print('start heavy search')
    model = gp.Model("mip1")
    x = model.addVars(30, name = 'x', vtype = GRB.BINARY)
    model.setObjective(40.78-sum(x[i*5+j]*topk1[i,j] for i in range(6)for j in range(5)))
    model.addConstr(x[0] + x[1] + x[2] + x[3] + x[4] == 1, "c0")
    model.addConstr(x[5] + x[6] + x[7] + x[8] + x[9] == 1, "c1")
    model.addConstr(x[10] + x[11] + x[12] + x[13] + x[14] == 1, "c2")
    model.addConstr(x[15] + x[16] + x[17] + x[18] + x[19] == 1, "c3")
    model.addConstr(x[20] + x[21] + x[22] + x[23] + x[24] == 1, "c4")
    model.addConstr(x[25] + x[26] + x[27] + x[28] + x[29] == 1, "c5")
    model.addConstr(lat_base-sum(x[i*5+j]*latency[i,j] for i in range(6)for j in range(5))<= latencylimit, "c6")
    model.addConstr(eng_base-sum(x[i*5+j]*energy[i,j] for i in range(6)for j in range(5))<= energylimit, "c7")
    model,accnew=search(model)
    
    archlast=[]
    for v in model.getVars():
        archlast.append(v.x)
    archnow=[]
    for i in range(6):
        dex=archlast[i*5:i*5+5]
        archnow.append(dex.index(1))
     
    ARCHcode=nasbench201_decode(archnow)
    print('The search net is',ARCHcode) 
    latencynew = lat_base-sum(latency[i,archnow[i]] for i in range(6))
    energynew = eng_base-sum(energy[i,archnow[i]] for i in range(6))
    print('The latency of search net is {:.1f} s'.format(latencynew))
    print('The energy of search net is {:.1f} s'.format(energynew))
    return ARCHcode, accnew, latencynew, energynew
