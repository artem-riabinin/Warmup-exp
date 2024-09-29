import matplotlib.pyplot as plt
import numpy as np
import random, sys, os
import asyncio, time
import threading
from datetime import datetime
import argparse

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader

# Utils
import utils
import compressors
from utils import add_params, sub_params, mult_param, norm_of_param

#=========================================================================================================
class NNConfiguration: pass
class WorkersConfiguration: pass
#=========================================================================================================
transfered_bits_by_node = None
fi_grad_calcs_by_node = None
train_loss = None
test_loss = None
train_acc = None
test_acc = None
fn_train_loss_grad_norm = None
fn_test_loss_grad_norm = None
#=========================================================================================================

print_lock = threading.Lock()

def dbgprint(wcfg, *args):
    printing_dbg = True
    if printing_dbg == True:
        print_lock.acquire()
        print(f"Worker {wcfg.worker_id}/{wcfg.total_workers}:", *args, flush = True)
        print_lock.release()

def rootprint(*args):
    print_lock.acquire()
    print(f"Master: ", *args, flush = True)
    print_lock.release()

def getAccuracy(model, trainset, batch_size, device):
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False,  pin_memory=False)
    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    accuracy = correct_predictions / total_samples
    model.train(prev_train_mode)
    return accuracy

def getLossAndGradNorm(model, trainset, batch_size, device):
    total_loss = 0
    grad_norm = 0
    one_inv_samples = torch.Tensor([1.0/len(trainset)]).to(device)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False)
    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)
    for p in model.parameters():
        p.grad = None
    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             
        logits = model(inputs)                                                              
        loss = one_inv_samples * F.cross_entropy(logits, outputs, reduction='sum')         
        loss.backward()                                                                     
        total_loss += loss
    for p in model.parameters(): 
        grad_norm += torch.norm(p.grad.data.flatten(0))**2
        p.grad = None
    model.train(prev_train_mode)
    return total_loss, grad_norm
    
#=========================================================================================================
class WorkerThread(threading.Thread):
  def __init__(self, wcfg, ncfg):
    super().__init__()
    self.wcfg = wcfg
    self.ncfg = ncfg
    self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
    self.model = self.model.to(wcfg.device)
    wcfg.model = self.model
 
  def run(self):
    wcfg = self.wcfg    # wcfg - configuration specific for worker
    ncfg = self.ncfg     # ncfg - general configuration with task description

    global transfered_bits_by_node, fi_grad_calcs_by_node
    global train_loss, test_loss, fn_train_loss_grad_norm, fn_test_loss_grad_norm

    dbgprint(wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ", len(wcfg.train_set))

    model = self.model

    loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders

    one_div_trainset_len    = torch.Tensor([1.0/len(wcfg.train_set)]).to(wcfg.device)
    one_div_batch_prime_len = torch.Tensor([1.0/(ncfg.batch_size_for_worker)]).to(wcfg.device)

    delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
    
    prev_train_mode = torch.is_grad_enabled()  
    model.train(True)

    for inputs, outputs in wcfg.train_loader:
        inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   
        logits = model(inputs)                                                             
        loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')     
        loss.backward()                                                                    
    g_i = []
        
    for p in model.parameters(): 
        g_i.append(p.grad.data.detach().clone())
        p.grad = None

    model.train(prev_train_mode)
    
    iteration = 0

    while True:
        wcfg.input_cmd_ready.acquire()
        if wcfg.cmd == "exit":
            wcfg.output_of_cmd = ""
            wcfg.cmd_output_ready.release()
            break

        if wcfg.cmd == "bcast_g_c0":
            wcfg.output_of_cmd = []

            # Generate subsample with b' cardinality
            indicies = wcfg.input_for_cmd[1]
            subset = torch.utils.data.Subset(wcfg.train_set, indicies)   
            minibatch_loader = DataLoader(subset, batch_size=ncfg.batch_size, shuffle=False, drop_last=False, pin_memory=False, collate_fn=None)

            # Evaluate in the previous point SGD within b' batch 
            prev_train_mode = torch.is_grad_enabled()  
            model.train(True)
            # Change xk: move from x(k) to x(k+1)
            k = 0
            for p in model.parameters(): 
                p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[0][k]
                k = k + 1

            # Evaluate SGD in the new point within b' batch 
            for inputs, outputs in minibatch_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   
                logits = model(inputs)                                                              
                loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs, reduction='sum')     
                loss.backward()                                                               
            g_batch_next = []
            for p in model.parameters():
                g_batch_next.append(p.grad.data.detach().clone())
                p.grad = None

            delta = sub_params(g_batch_next, g_i)

            delta_offset = 0
            for t in range(len(delta)):
                offset = len(delta[t].flatten(0))
                delta_flatten[(delta_offset):(delta_offset + offset)] = delta[t].flatten(0)
                delta_offset += offset

            delta_flatten = wcfg.compressor.compressVector(delta_flatten)

            delta_offset = 0
            for t in range(len(delta)):
                offset = len(delta[t].flatten(0))
                delta[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
                delta_offset += offset

            g_i = add_params(g_i, delta)
            wcfg.output_of_cmd = g_i
            model.train(prev_train_mode)

            transfered_bits_by_node[wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size
            fi_grad_calcs_by_node[wcfg.worker_id, iteration] = ncfg.batch_size_for_worker
            iteration += 1

            wcfg.cmd_output_ready.release()

        if wcfg.cmd == "full_grad":
            wcfg.output_of_cmd = []
            prev_train_mode = torch.is_grad_enabled()  
            model.train(True)

            for inputs, outputs in wcfg.train_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)     
                logits = model(inputs)                                                
                loss = one_div_trainset_len * F.cross_entropy(logits, outputs)        
                loss.backward()                                                       

            for p in model.parameters(): 
                wcfg.output_of_cmd.append(p.grad.data.detach().clone())
                p.grad = None

            model.train(prev_train_mode)
            wcfg.cmd_output_ready.release()

    dbgprint(wcfg, f"END")
#=========================================================================================================

def main():
    global transfered_bits_by_node
    global fi_grad_calcs_by_node
    global train_loss
    global test_loss
    global train_acc
    global test_acc
    global fn_train_loss_grad_norm
    global fn_test_loss_grad_norm
    
    #=======================================================================================================
    parser = argparse.ArgumentParser(description='Run top-k algorithm')
    parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=3000, help='Maximum number of iteration')
    parser.add_argument('--k', action='store', dest='k', type=int, default=100000, help='Sparcification parameter')
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=5, help='Number of workers that will be used')
    parser.add_argument('--factor', action='store', dest='factor', type=int, default=1, help='Stepsize factor')
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=1024, help='Batch size per worker and for GPU')
    parser.add_argument('--model', action='store', dest='model', type=str, default='resnet20', help='Name of NN architechture')
    parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='CIFAR10', help='Name of dataset')
    args = parser.parse_args()
    #=======================================================================================================
    
    utils.printTorchInfo()
    print("******************************************************************")
    cpu_device   = torch.device("cpu")      # CPU device
    gpu_device_0 = torch.device('cuda:0')   # Selected GPU (index 0)

    available_devices = [gpu_device_0]
    master_device = available_devices[0]
    print("******************************************************************")
    global nn_config, workers_config

    # Configuration for NN
    # SETTING SEED for reproducity
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nn_config = NNConfiguration()
    nn_config.dataset = args.dataset #"CIFAR10"                     # Dataset
    nn_config.model_name = args.model                                  # NN architecture, resnet20
    nn_config.load_workers = 0                                                 # How many subprocesses to use for data loading, 0 means that the data will be loaded in the main process.
    nn_config.batch_size = args.batch_size                              # Batch size for training, 128
    nn_config.KMax = args.max_it                                             # Maximum number of iterations
    kWorkers = args.num_workers                                             # Number of workers

    # Load data
    train_sets, train_set_full, test_set, train_loaders, test_loaders, classes = utils.getSplitDatasets(nn_config.dataset, nn_config.batch_size, nn_config.load_workers, kWorkers)

    print(f"Start training {nn_config.model_name}@{nn_config.dataset} for K={nn_config.KMax} iteration. EF21", available_devices)
    master_model = utils.getModel(nn_config.model_name, train_set_full, master_device)
    master_model = master_model.to(master_device)

    utils.printLayersInfo(master_model, nn_config.model_name)
    nn_config.D = utils.numberOfParams(master_model)
    nn_config.component_bits_size = 32

    # METRICS 
    transfered_bits_by_node = np.zeros((kWorkers, nn_config.KMax)) # Transfered bits
    fi_grad_calcs_by_node   = np.zeros((kWorkers, nn_config.KMax)) # Evaluate number gradients for fi
    train_loss = np.zeros((nn_config.KMax))                                          # Train loss
    test_loss = np.zeros((nn_config.KMax))                                           # Test loss
    train_acc = np.zeros((nn_config.KMax))                                           # Train accuracy
    test_acc = np.zeros((nn_config.KMax))                                            # Test accuracy
    fn_train_loss_grad_norm = np.zeros((nn_config.KMax))                  # Gradient norm for train loss
    fn_test_loss_grad_norm  = np.zeros((nn_config.KMax))                  # Gradient norm for test loss    

    # TUNABLE PARAMS
    #=======================================================================================================
    K = args.k #27000
    gamma = args.factor*1e-3   #5000*1e-3
    batch_size_for_worker = args.batch_size   #128
    #=======================================================================================================
    nn_config.kWorkers = kWorkers
    nn_config.i_use_vr_biased_diana  = True 
    nn_config.train_set_full_samples = len(train_set_full)
    nn_config.train_sets_samples = [len(s) for s in train_sets]
    nn_config.test_set_samples = len(test_set)
    c_top = compressors.Compressor()
    c_top.makeTopKCompressor(int(K), nn_config.D)
    alpha = c_top.getAlpha() 
    nn_config.gamma = gamma                                    
    nn_config.batch_size_for_worker = batch_size_for_worker 

    worker_tasks = []                           # Worker tasks
    worker_cfgs = []                             # Worker configurations

    for i in range(kWorkers):
        worker_cfgs.append(WorkersConfiguration())
        worker_cfgs[-1].worker_id = i
        worker_cfgs[-1].total_workers = kWorkers
        worker_cfgs[-1].train_set = train_sets[i]
        worker_cfgs[-1].test_set = test_set
        worker_cfgs[-1].train_set_full = train_set_full
        worker_cfgs[-1].train_loader = train_loaders[i]
        worker_cfgs[-1].test_loader = test_loaders[i]
        worker_cfgs[-1].classes = classes
        worker_cfgs[-1].device = available_devices[i % len(available_devices)]              
        worker_cfgs[-1].compressor = compressors.Compressor()
        worker_cfgs[-1].compressor.makeTopKCompressor(int(K), nn_config.D) 
        worker_cfgs[-1].input_cmd_ready  = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd_output_ready = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd = "init"
        worker_cfgs[-1].input_for_cmd = ""
        worker_cfgs[-1].output_of_cmd = ""
        worker_tasks.append(WorkerThread(worker_cfgs[-1], nn_config))

    for i in range(kWorkers):
        worker_tasks[i].start()
    #=======================================================================================================
    # Evaluate g0
    for i in range(kWorkers):
        worker_cfgs[i].cmd = "full_grad"
        worker_cfgs[i].input_cmd_ready.release()

    for i in range(kWorkers):
        worker_cfgs[i].cmd_output_ready.acquire()

    g0 = worker_cfgs[0].output_of_cmd
    worker_cfgs[0].output_of_cmd = None
    for i in range(1, kWorkers):
        for j in range(len(worker_cfgs[i].output_of_cmd)):
            g0[j] = g0[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
        worker_cfgs[i].output_of_cmd = None
    g0 = mult_param(1.0/kWorkers, g0)
    gk = g0

    rootprint(f"Start {nn_config.KMax} iterations of algorithm")

    for iteration in range(0, nn_config.KMax):
        rootprint(f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration/nn_config.KMax * 100.0, "%")
    #=======================================================================================================
    #Collecting Statistics
        if iteration % 50 == 0:
            loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
            train_loss[iteration] = loss
            fn_train_loss_grad_norm[iteration] = grad_norm
            loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
            test_loss[iteration] = loss
            fn_test_loss_grad_norm[iteration] = grad_norm
            train_acc[iteration] = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
            test_acc[iteration]  = getAccuracy(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
            print(f"  train accuracy: {train_acc[iteration]}, test accuracy: {test_acc[iteration]}, train loss: {train_loss[iteration]}, test loss: {test_loss[iteration]}")
            print(f"  grad norm train: {fn_train_loss_grad_norm[iteration]}, test: {fn_test_loss_grad_norm[iteration]}")
            print(f"  used step-size: {nn_config.gamma}")
        else:
            train_loss[iteration] = train_loss[iteration - 1] 
            fn_train_loss_grad_norm[iteration] = fn_train_loss_grad_norm[iteration - 1]
            test_loss[iteration] = test_loss[iteration - 1]
            fn_test_loss_grad_norm[iteration]  = fn_test_loss_grad_norm[iteration - 1] 
            train_acc[iteration] = train_acc[iteration - 1]
            test_acc[iteration] = test_acc[iteration - 1]
    #=======================================================================================================
        gk_for_device = {}
        for d_id in range(len(available_devices)):
            gk_loc = []
            for gk_i in gk:
                gk_loc.append(gk_i.to(available_devices[d_id]))
            gk_for_device[available_devices[d_id]] = gk_loc

        for i in range(kWorkers):
            worker_cfgs[i].cmd = "bcast_g_c0"
            worker_cfgs[i].input_for_cmd = [gk_for_device[worker_cfgs[i].device], torch.randperm(len(train_sets[i]))[:batch_size_for_worker].to(worker_cfgs[i].device)]
            worker_cfgs[i].input_cmd_ready.release()
                
        for i in range(kWorkers): 
            worker_cfgs[i].cmd_output_ready.acquire()

        gk_next = worker_cfgs[0].output_of_cmd
        worker_cfgs[0].output_of_cmd = None
        for i in range(1, kWorkers): 
            for j in range(len(worker_cfgs[i].output_of_cmd)):
                gk_next[j] = gk_next[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
            worker_cfgs[i].output_of_cmd = None
        gk_next = mult_param(1.0/kWorkers, gk_next)
        gk = gk_next
    #=======================================================================================================
    # Finish all work of nodes
    for i in range(kWorkers):
        worker_cfgs[i].cmd = "exit"
        worker_cfgs[i].input_cmd_ready.release()
    for i in range(kWorkers):
        worker_tasks[i].join()
    print(f"Master has been finished")
    my = {}
    my["transfered_bits_by_node"] = transfered_bits_by_node
    my["fi_grad_calcs_by_node"] = fi_grad_calcs_by_node
    my["train_loss"] = train_loss
    my["test_loss"] = test_loss
    my["train_acc"] = train_acc
    my["test_acc"]  = test_acc
    my["fn_train_loss_grad_norm"] = fn_train_loss_grad_norm
    my["fn_test_loss_grad_norm"] = fn_test_loss_grad_norm
    my["nn_config"] = nn_config
    my["current_data_and_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    my["experiment_description"] = f"Training {nn_config.model_name}@{nn_config.dataset}"
    my["compressors"] = worker_cfgs[0].compressor.fullName()
    my["algo_name"] = f"EF21"
    if hasattr(worker_cfgs[0].compressor, "K"):    
        my["K"] = worker_cfgs[0].compressor.K
    prefix4algo = "EF21"
    ser_fname = f"experiment_{prefix4algo}_K_{K}_sz_{int(gamma/1e-3)}_nw_{kWorkers}_bsz_{batch_size_for_worker}_{nn_config.model_name}_at_{nn_config.dataset}.bin"
    utils.serialize(my, ser_fname)
    print(f"Experiment info has been serialised into '{ser_fname}'")
    #=======================================================================================================

if __name__ == "__main__":
    main()
