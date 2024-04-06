#!/usr/bin/env python

print("Start at the beginning of training!")
import io, os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import copy
import h5py
import torch.optim as optim
import argparse

import model as mdp
import util
import compute_kernel as ker

print("torch, get_num_threads() = ", torch.get_num_threads())
print("torch, get_num_interop_threads() = ", torch.get_num_interop_threads())

print("Start setting parser!", flush=True)

#----------------------Parser settings---------------------------

parser = argparse.ArgumentParser(description='Exalearn_Training_v1')

parser.add_argument('--batch_size',     type=int,   default=8192,
                    help='input batch size for training (default: 8192)')
parser.add_argument('--epochs',         type=int,   default=4000,
                    help='number of epochs to train (default: 4000)')
parser.add_argument('--lr',             type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random seed (default: 42)')
parser.add_argument('--log_interval',   type=int,   default=1,
                    help='how many batches to wait before logging training status')
parser.add_argument('--blind_train_epoch',   type=int,   default=1000,
                    help='number of epochs to train without early stopping (default: 1000)')
parser.add_argument('--device',         default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--ntrain',         type=float, default=2000,
                    help='number of training samples (default: 2000)')
parser.add_argument('--ntest',          type=float, default=1000,
                    help='number of test samples (default: 500)')
parser.add_argument('--num_workers',    type=int,   default=1, 
                    help='set the number of op workers. only work for gpu')
parser.add_argument('--data_dir', default='./',
                    help='root directory of base/test/study/AL subdir')

args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )

if args.cuda:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if not args.cuda:
    torch.use_deterministic_algorithms(True)

def print_memory_usage(info):
    util.print_memory_usage_template("train_step1", info)

print_memory_usage("Starting!")

#-----------------------------Loading data--------------------------------

base_cubic_file       = os.path.join(args.data_dir, "base/data/cubic_1001460_cubic.hdf5")
base_trigonal_file    = os.path.join(args.data_dir, "base/data/trigonal_1522004_trigonal.hdf5")
base_tetragonal_file  = os.path.join(args.data_dir, "base/data/tetragonal_1531431_tetragonal.hdf5")
test_cubic_file       = os.path.join(args.data_dir, "test/data/cubic_1001460_cubic.hdf5")
test_trigonal_file    = os.path.join(args.data_dir, "test/data/trigonal_1522004_trigonal.hdf5")
test_tetragonal_file  = os.path.join(args.data_dir, "test/data/tetragonal_1531431_tetragonal.hdf5")
study_cubic_file      = os.path.join(args.data_dir, "study/data/cubic_1001460_cubic.hdf5")
study_trigonal_file   = os.path.join(args.data_dir, "study/data/trigonal_1522004_trigonal.hdf5")
study_tetragonal_file = os.path.join(args.data_dir, "study/data/tetragonal_1531431_tetragonal.hdf5")

x_train,  y_train  = util.create_numpy_data(base_cubic_file, base_trigonal_file,  base_tetragonal_file)
x_test,  y_test  = util.create_numpy_data(test_cubic_file, test_trigonal_file,  test_tetragonal_file)
x_study, y_study = util.create_numpy_data(study_cubic_file, study_trigonal_file, study_tetragonal_file)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_study.shape)
print(y_study.shape)

print_memory_usage("Finish loading data!")

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
x_study_torch = torch.from_numpy(x_study).float()
x_study_torch = x_study_torch.reshape((x_study_torch.shape[0], 1, x_study_torch.shape[1]))
y_study_torch = torch.from_numpy(y_study).float()
print(x_study_torch.shape)
print(y_study_torch.shape)

torch.save(x_study_torch, "x_study_torch.pt")

x_train_torch = torch.from_numpy(x_train).float()
x_train_torch = x_train_torch.reshape((x_train_torch.shape[0], 1, x_train_torch.shape[1]))
y_train_torch = torch.from_numpy(y_train).float()
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(x_train_torch.shape)
print(y_train_torch.shape)

x_test_torch = torch.from_numpy(x_test).float()
x_test_torch = x_test_torch.reshape((x_test_torch.shape[0], 1, x_test_torch.shape[1]))
y_test_torch = torch.from_numpy(y_test).float()
test_dataset = torch.utils.data.TensorDataset(x_test_torch, y_test_torch)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(x_test_torch.shape)
print(y_test_torch.shape)

print_memory_usage("Finish creating torch dataset and loader!")

#----------------------------setup model---------------------------------
#Important! FIXME
#Here num_output should be 3+1 instead of 3, since each sample needs one value representing its uncertainty
model = mdp.FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model = model.cuda()

print_memory_usage("Finish creating model!")

#---------------------------setup optimizer------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def criterion_reg(y_pred, y_true):
    y_pred_value = y_pred[:, 0:3].reshape(-1, 3)
    logsig2 = y_pred[:, 3].reshape(-1, 1)
    l2_diff = torch.sum((y_true - y_pred_value) ** 2, axis=1, keepdims=True) / torch.exp(logsig2) + logsig2
    return torch.mean(l2_diff)

#criterion_class = torch.nn.BCEWithLogitsLoss()
criterion_class = torch.nn.CrossEntropyLoss()

def lr_lambda(epoch):
    if epoch <= 5000:
        return 1.0
    elif 5001 <= epoch <= 10000:
        return 0.5
    else:
        return 0.2

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print_memory_usage("finish setting up optimizer!")

#------------------------------start training----------------------------------

train_loss_list = []
test_loss_list = []
early_stopping = util.EarlyStopping(max_num=100//args.log_interval, min_delta=0.001)

time_tot = time.time()

for epoch in range(0, args.epochs):

    ker.train(epoch, 
          model = model,
          optimizer = optimizer,
          train_loader = train_loader,
          criterion_reg = criterion_reg, criterion_class = criterion_class,
          lr_scheduler = scheduler,
          on_gpu = args.cuda,
          log_interval = args.log_interval,
          train_loss_list)

    test_loss = ker.test(epoch, 
                     model = model, 
                     test_loader = test_loader,
                     criterion_reg = criterion_reg, criterion_class = criterion_class,
                     on_gpu = args.cuda,
                     log_interval = args.log_interval,
                     test_loss_list)

    if epoch > args.blind_train_epoch:
        early_stopping(test_loss)
        if early_stopping.improve:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, 'ckpt.pth')
            print("Save model at epoch ", epoch)
        if early_stopping.do_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break;

print("Best val loss = ", early_stopping.best_loss)
time_tot = time.time() - time_tot
print("Total training time = {}".format(time_tot))

print_memory_usage("finish first training part")

checkpoint = torch.load('ckpt.pth')
model = mdp.FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
model.load_state_dict(checkpoint['model_state_dict'])

print_memory_usage("finish loading from disk")

l2_diff, sigma2, class_loss = validation(model, x_test_torch, y_test_torch, criterion_class)
