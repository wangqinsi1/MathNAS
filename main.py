import os
import torch
import argparse
from dataprocess import supervit_data, supertransformer_data, mobilenetv3_data, nasbench201_data
from search import supervit_search, supertransformer_search, mobilenetv3_search, nasbench201_search
from predict import supervit_predict, supertransformer_predict, mobilenetv3_predict, nasbench201_predict
import yaml

 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='predict|nas|valid')
    parser.add_argument('--search_space', type=str, default='supervit', help='nasbench201|mobilenet|supertransformer|supervit')
    parser.add_argument('--device', type=str, default=None, help="raspberry pi, xeon, titan xp")
    parser.add_argument('--main_path', type=str, default='.')
    parser.add_argument('--save_path', type=str, default='results', help='')
    parser.add_argument('--load_path', type=str, default=None, help='path that load predicted net')
    parser.add_argument('--latency_constraint', type=float, default=None, help="latency constraint when performing NAS process")
    parser.add_argument('--energy_constraint', type=float, default=500000, help="energy constraint when performing NAS process")
    parser.add_argument('--flops_constraint', type=float, default=None, help="flops constraint when performing NAS process")
    args = parser.parse_args()
    return args

def set_path(args):
    args.data_path = os.path.join(
        args.main_path, 'data', args.search_space)
    args.save_path = os.path.join(
            args.save_path, args.search_space)  
    if args.device:
        args.save_path = os.path.join(args.save_path, args.device) 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path) 
    print(f'==> save path is [{args.save_path}] ...')   
    return args 


def main():
    args = get_parser()
    args = set_path(args)
    print(f'=====> search space is [{args.search_space}] ...')
    if args.mode == 'predict':
        if args.load_path != None:
            if args.search_space == 'supervit':
                block_acc,block_flop=supervit_data()
                acc_predict,flop_predict = supervit_predict(block_acc,block_flop,args.load_path)
            if args.search_space == 'supertransformer':  
                if args.device != None:
                    block_acc,block_lat=supertransformer_data(args.device)
                    acc_predict,lat_predict = supertransformer_predict(block_acc,block_lat,args.device,args.load_path)
                else:
                    print(f'Error! Please set the device.')
            if args.search_space == 'mobilenetv3':  
                if args.device != None:
                    block_acc,block_lat = mobilenetv3_data(args.device)
                    acc_predict,lat_predict = mobilenetv3_predict(block_acc,block_lat,args.device,args.load_path)
                else:
                    print(f'Error! Please set the device.')  
            if args.search_space == 'nasbench201':  
                if args.device != None:
                    block_acc,block_lat,block_eng = nasbench201_data(args.device)
                    acc_predict,lat_predict,eng_predict = nasbench201_predict(block_acc,block_lat,block_eng,args.device,args.load_path)
                else:
                    print(f'Error! Please set the device.')     
                    
        else:
            print(f'Error! Please set network load path.')
            
    if args.mode == 'nas':
        if args.search_space == 'supervit':
            if args.flops_constraint != None:
                print(f'=====> FLOPs limit is {args.flops_constraint} M ...')
                block_acc,block_flop=supervit_data()
                arch,arch_acc,arch_flops=supervit_search(block_acc,block_flop,args.flops_constraint)
                print(arch)
                filename=args.save_path+'/'+str(args.flops_constraint)+'.yml'
                with open(filename, 'w') as file:
                    yaml.dump(arch, file)
                print(f'network saved!')
            else:
                print(f'Error! Please set flops limit of network.')

        if args.search_space == 'supertransformer':
            if args.latency_constraint!= None and args.device != None:
                block_acc,block_lat=supertransformer_data(args.device)
                arch,arch_acc,arch_lat=supertransformer_search(block_acc,block_lat,args.latency_constraint,args.device)
                filename=args.save_path+'/'+str(args.latency_constraint)+'.yml'
                with open(filename, 'w') as file:
                    yaml.dump(arch, file)
                print(f'network saved!')
            else:
                print(f'Error! Please set latency limit of network and the device.')
          
        
        if args.search_space == 'mobilenetv3':
            if args.latency_constraint!= None and args.device != None:
                block_acc,block_lat= mobilenetv3_data(args.device)
                arch,arch_acc,arch_lat=mobilenetv3_search(block_acc, block_lat, args.latency_constraint, args.device)
                filename=args.save_path+'/'+str(args.latency_constraint)+'.yml'
                with open(filename, 'w') as file:
                    yaml.dump(arch, file)
                print(f'network saved!')
            else:
                print(f'Error! Please set latency limit of network and the device.')
             
        if args.search_space == 'nasbench201':
            if args.latency_constraint!= None and args.energy_constraint!= None and args.device != None:
                block_acc,block_lat,block_eng= nasbench201_data(args.device)
                arch,arch_acc,arch_lat,arch_eng=nasbench201_search(block_acc, block_lat,block_eng, args.latency_constraint,args.energy_constraint, args.device)
                filename=args.save_path+'/'+str(args.latency_constraint)+'_'+str(args.energy_constraint)+'.yml'
                with open(filename, 'w') as file:
                    yaml.dump(arch, file)
                print(f'network saved!')
            else:
                print(f'Error! Please set latency limit and energy limit of network and the device.')



if __name__ == '__main__':
    main()

