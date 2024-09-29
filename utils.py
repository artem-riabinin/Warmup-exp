import numpy as np
import matplotlib.pyplot as plt

import sys, os, pickle, math
import psutil 
import time

import torch
import torchvision

import torchvision.transforms as transforms

from resnet import resnet20

from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path

import_time = time.time()

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 27

def serialize(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def deserialize(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def printSystemInfo():
    print("")
    print("*********************************************************************************************")
    print("Path to python interpretator:", sys.executable)
    print("Version:", sys.version)
    print("Platform name:", sys.platform)
    print("Physical CPU processors: ", psutil.cpu_count(logical=False))
    print("Logical CPU processors: ", psutil.cpu_count(logical=True))
    print("Current CPU Frequncy: ", psutil.cpu_freq().current, "MHz")
    print("Installed Physical available RAM Memory: %g %s" % (psutil.virtual_memory().total/(1024.0**3), "GBytes"))
    print("Available physical available RAM Memory: %g %s" % (psutil.virtual_memory().available/(1024.0**3), "GBytes"))
    print("")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print("Process Resident Set (Working Set) Size: ", mem_info.rss/(1024.0 * 1024.0), "MBytes")
    print("Virtual Memory used by process: ", mem_info.vms/(1024.0 * 1024.0), "MBytes")
    print("*********************************************************************************************")
    print("Time since program starts: ", str(time.time() - import_time), " seconds")
    print("*********************************************************************************************")
    print("Script name: ", sys.argv[0])
    print("*********************************************************************************************")

def printTorchInfo():
    print("******************************************************************")
    print("Is CUDA avilable:", torch.cuda.is_available())
    print("GPU devices: ", torch.cuda.device_count())
    print("Torch hub(cache) for loaded datasets: ", torch.hub.get_dir())
    print("******************************************************************")
    print("")
    print(get_pretty_env_info())
    print("******************************************************************")

def numberOfParams(model):
    total_number_of_scalar_parameters = 0
    for p in model.parameters(): 
        total_items_in_param = 1
        for i in range(p.data.dim()):
            total_items_in_param = total_items_in_param * p.data.size(i)
        total_number_of_scalar_parameters += total_items_in_param
    return total_number_of_scalar_parameters

def printLayersInfo(model,model_name):
    # Statistics about used modules inside NN
    max_string_length = 0
    basic_modules = {}

    for module in model.modules():
        class_name = str(type(module)).replace("class ", "").replace("<'", "").replace("'>", "")
        if class_name.find("torch.nn") != 0:
            continue
        max_string_length = max(max_string_length, len(class_name))
 
        if class_name not in basic_modules:
            basic_modules[class_name]  = 1
        else:
            basic_modules[class_name] += 1

    print(f"Summary about layers inside {model_name}")
    print("=============================================================")
    for (layer, count) in basic_modules.items():
        print(f"{layer:{max_string_length + 1}s} occured {count:02d} times")
    print("=============================================================")
    print("Total number of parameters inside '{}' is {:,}".format(model_name, numberOfParams(model)))
    print("=============================================================")
    
#======================================================================================================= 

def add_params(x, y):
    return [xi + yi for xi, yi in zip(x, y)]

def sub_params(x, y):
    return [xi - yi for xi, yi in zip(x, y)]

def mult_param(alpha, x):
    return [alpha * xi for xi in x]

def norm_of_param(x):
    return sum(torch.norm(param.flatten()) for param in x)
    
#=======================================================================================================      

def getModel(model_name, dataset, device):
    model = resnet20(num_classes=10)
    model = model.to(device)
    model.train(False)

    return model

def getSplitDatasets(dataset_name, batch_size, load_workers, train_workers):
    root_dir  = Path(torch.hub.get_dir()) / f'datasets/{dataset_name}'
    ds = getattr(torchvision.datasets, dataset_name)
    transform = transforms.Compose([
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape 
                # (C x H x W) in the range [0.0, 1.0]  
                # https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.ToTensor
                transforms.ToTensor(),
                #  https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Normalize
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_set = ds(root=root_dir, 
                   train=True, 
                   download=True, 
                   transform = transform
                  )

    test_set = ds(root=root_dir, 
                  train=False, 
                  download=True, 
                  transform = transform
                  )

    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    #==================================================================================================================================   
    classes = None

    if dataset_name == "CIFAR10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_sets, train_set, test_set, train_loaders, test_loaders, classes 