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
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import argparse

print("torch, get_num_threads() = ", torch.get_num_threads())
print("torch, get_num_interop_threads() = ", torch.get_num_interop_threads())

print("Start setting parser!", flush=True)

#----------------------Parser settings---------------------------

parser = argparse.ArgumentParser(description='Exalearn_AL_v1')

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
parser.add_argument('--phase_idx',    type=int, 
                    help='which AL phase we are in. This is one-indexed! In other word, first running this script means idx=1')
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

#------------------------Model----------------------------
class FullModel(torch.nn.Module):
    def __init__(self, len_input, num_hidden, num_output,
                 conv1=(16, 3, 1), 
                 pool1=(2, 2), 
                 conv2=(32, 4, 2), 
                 pool2=(2, 2), 
                 fc1=256, 
                 num_classes=3):
        super(FullModel, self).__init__()
        
        n = len_input
        # In-channels, Out-channels, Kernel_size, stride ...
        self.conv1 = torch.nn.Conv1d(1, conv1[0], conv1[1], stride=conv1[2])
        n = (n - conv1[1]) // conv1[2] + 1

        self.pool1 = torch.nn.MaxPool1d(pool1[0], stride=pool1[1] )
        n = (n - pool1[0]) // pool1[1] + 1
        
        self.conv2 = torch.nn.Conv1d(conv1[0], conv2[0], conv2[1], stride=conv2[2])
        n = (n - conv2[1]) // conv2[2] + 1
        
        self.pool2 = torch.nn.MaxPool1d(pool2[0], stride=pool2[1] )
        n = (n - pool2[0]) // pool2[1] + 1

        self.relu = torch.nn.LeakyReLU(0.1)
        self.features = torch.nn.Sequential( self.conv1, self.relu, self.pool1, self.conv2, self.relu, self.pool2 )
        self.fc1 = torch.nn.Linear(n*conv2[0], fc1)
        self.fc2 = torch.nn.Linear(fc1, num_classes)        
        self.regression_layer=torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        class_output = self.fc2(x)
        regression_output = self.regression_layer(x)
        return class_output, regression_output


#-----------------------------Loading data--------------------------------
def create_numpy_data(file_cubic, file_trigonal, file_tetragonal):
    with h5py.File(file_cubic, 'r') as f:
        dhisto = f['histograms']
        x_cubic = dhisto[:, 1, :]
        x_shape = x_cubic.shape
        dparams = f['parameters']
        y_cubic = dparams[:]
        y_shape = y_cubic.shape
        print(x_shape)
        print(y_shape)
    with h5py.File(file_trigonal, 'r') as f:
        dhisto = f['histograms']
        x_trigonal = dhisto[:, 1, :]
        x_shape = x_trigonal.shape
        dparams = f['parameters']
        y_trigonal = dparams[:]
        y_shape = y_trigonal.shape
        print(x_shape)
        print(y_shape)
    with h5py.File(file_tetragonal, 'r') as f:
        dhisto = f['histograms']
        x_tetragonal = dhisto[:, 1, :]
        x_shape = x_tetragonal.shape
        dparams = f['parameters']
        y_tetragonal = dparams[:]
        y_shape = y_tetragonal.shape
        print(x_shape)
        print(y_shape)
    
    x = np.concatenate([x_cubic, x_trigonal, x_tetragonal], axis=0)
    scaler_x = MinMaxScaler(copy=True)
    x = scaler_x.fit_transform(x.T).T

    y = np.concatenate([y_cubic, y_trigonal, y_tetragonal], axis=0)
    y[:,0] = (y[:,0] - 3.8 )  / 0.4
    y[:,1] = (y[:,1] - 3.8 )  / 0.4
    y[:,2] = (y[:,2] - 60.0 ) / 60.0

    return x, y

base_cubic_file       = os.path.join(args.data_dir, "base/data/cubic_1001460_cubic.hdf5")
base_trigonal_file    = os.path.join(args.data_dir, "base/data/trigonal_1522004_trigonal.hdf5")
base_tetragonal_file  = os.path.join(args.data_dir, "base/data/tetragonal_1531431_tetragonal.hdf5")
test_cubic_file       = os.path.join(args.data_dir, "test/data/cubic_1001460_cubic.hdf5")
test_trigonal_file    = os.path.join(args.data_dir, "test/data/trigonal_1522004_trigonal.hdf5")
test_tetragonal_file  = os.path.join(args.data_dir, "test/data/tetragonal_1531431_tetragonal.hdf5")
study_cubic_file      = os.path.join(args.data_dir, "study/data/cubic_1001460_cubic.hdf5")
study_trigonal_file   = os.path.join(args.data_dir, "study/data/trigonal_1522004_trigonal.hdf5")
study_tetragonal_file = os.path.join(args.data_dir, "study/data/tetragonal_1531431_tetragonal.hdf5")

x_orig,  y_orig  = create_numpy_data(base_cubic_file, base_trigonal_file,  base_tetragonal_file)
x_test,  y_test  = create_numpy_data(test_cubic_file, test_trigonal_file,  test_tetragonal_file)
x_study, y_study = create_numpy_data(study_cubic_file, study_trigonal_file, study_tetragonal_file)

#Need to shuffle! 
#Temporarily reset seed to a fixed value. This should never be changed in the same complete run to make sure we always have the same dataset!
np.random.seed(42)
shuffled_indices = np.random.permutation(x_orig.shape[0])
x_orig = x_orig[shuffled_indices]
y_orig = y_orig[shuffled_indices]
#Then we set seed back to our choice
np.random.seed(args.seed)


x_train = x_orig[0:args.ntrain]
y_train = y_orig[0:args.ntrain]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_study.shape)
print(y_study.shape)


kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
x_study_torch = torch.from_numpy(x_study).float()
x_study_torch = x_study_torch.reshape((x_study_torch.shape[0], 1, x_study_torch.shape[1]))
y_study_torch = torch.from_numpy(y_study).float()
print(x_study_torch.shape)
print(y_study_torch.shape)

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

#----------------------------setup model---------------------------------
#Important! FIXME
#Here num_output should be 3+1 instead of 3, since each sample needs one value representing its uncertainty
model = FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model.cuda()

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

class EarlyStopping:
    def __init__(self, max_num, min_delta):
        self.max_num = max_num
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 99999999.9
        self.do_stop = False
        self.improve = False

    def __call__(self, loss):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.improve = True
        else:
            self.counter += 1
            if self.counter > self.max_num:
                self.do_stop = True
            self.improve = False


#------------------------------start training----------------------------------

def train(epoch, loss_list):
    model.train()
    print(optimizer.param_groups[0]['lr'])

    if epoch % args.log_interval == 0:
        running_loss  = torch.tensor(0.0)
        running_loss1 = torch.tensor(0.0)
        running_loss2 = torch.tensor(0.0)
        if args.cuda:
            running_loss, running_loss1, running_loss2  = running_loss.cuda(), running_loss1.cuda(), running_loss2.cuda()

    for batch_idx, current_batch in enumerate(train_loader):
        if args.cuda:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]

        optimizer.zero_grad()
        class_output, regression_output = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if args.cuda:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        loss1 = criterion_reg(regression_output, regression_gndtruth)
        loss2 = criterion_class(class_output, class_gndtruth)
        loss  = loss1 + loss2
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            running_loss  += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        scheduler.step()

    if epoch % args.log_interval == 0:
        running_loss  = running_loss  / len(train_loader)
        running_loss1 = running_loss1 / len(train_loader)
        running_loss2 = running_loss2 / len(train_loader)
        print("epoch: {}, Average loss_reg: {:15.8f}, loss_class: {:15.8f}, loss_tot: {:15.8f}".format(epoch, running_loss1, running_loss2, running_loss))
        loss_list.append(running_loss)

def test(epoch, loss_list):
    model.eval()
    
    test_loss  = torch.tensor(0.0)
    test_loss1 = torch.tensor(0.0)
    test_loss2 = torch.tensor(0.0)
    if args.cuda:
        test_loss, test_loss1, test_loss2  = test_loss.cuda(), test_loss1.cuda(), test_loss2.cuda()
    
    for batch_idx, current_batch in enumerate(test_loader):
        if args.cuda:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]        

        y_pred_torch_class, y_pred_torch_regression = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if args.cuda:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        test_loss1 += criterion_reg(y_pred_torch_regression, regression_gndtruth).item()
        test_loss2 += criterion_class(y_pred_torch_class, class_gndtruth).item()
    
    test_loss1 = test_loss1.detach().cpu().numpy() / len(test_loader)
    test_loss2 = test_loss2.detach().cpu().numpy() / len(test_loader)
    test_loss  = test_loss1 + test_loss2
    if epoch % args.log_interval == 0:
        print("epoch: {}, Average test_loss_reg: {:15.8f}, test_loss_class: {:15.8f}, test_loss_tot: {:15.8f}".format(epoch, test_loss1, test_loss2, test_loss))
    loss_list.append(test_loss)
    return test_loss

train_loss_list = []
test_loss_list = []
early_stopping = EarlyStopping(max_num=100, min_delta=0.001)


#------------------------first go with AL branch----------------------------
model = FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load('ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def validation():
    y_pred_torch_class, y_pred_torch_regression = model(x_test_torch.cuda())
    y_pred_torch_class = y_pred_torch_class.detach().cpu()
    y_pred_torch_regression = y_pred_torch_regression.detach().reshape(-1, 4).cpu()
    avg_diff_0 = torch.mean((y_pred_torch_regression[:,0:3] - y_test_torch[:,0:3]) ** 2).numpy()
    avg_sigma2 = torch.mean(torch.exp(y_pred_torch_regression[:,3])).numpy()
    avg_class_loss = criterion_class(y_pred_torch_class, y_test_torch[:,3].type(torch.LongTensor)).numpy()
    print("Avg diff on test set = ", avg_diff_0)
    print("Avg sigma^2 on test set = ", avg_sigma2)
    print("Avg class loss on test set = ", avg_class_loss)
    return avg_diff_0, avg_sigma2, avg_class_loss

#Here we do not evaluate the last model anymore and leave it to the last iteration
#l2_diff_0, sigma2_0, class_loss_0 = validation()

x_AL_list = []
y_AL_list = []
num_extra = 0

#In phase_0 (another script), we plan to create the first AL dataset, but not use that for training
#In phase_k, we already have base data and AL_1 upto AL_k
for i in range(1, args.phase_idx + 1):
    AL_cubic_file       = os.path.join(args.data_dir, "AL_phase_{}/data/cubic_1001460_cubic.hdf5".format(args.phase_idx))
    AL_trigonal_file    = os.path.join(args.data_dir, "AL_phase_{}/data/trigonal_1522004_trigonal.hdf5".format(args.phase_idx))
    AL_tetragonal_file  = os.path.join(args.data_dir, "AL_phase_{}/data/tetragonal_1531431_tetragonal.hdf5".format(args.phase_idx))
    x_AL_temp, y_AL_temp = create_numpy_data(AL_cubic_file, AL_trigonal_file, AL_tetragonal_file)
    x_AL_list.append(x_AL_temp)
    y_AL_list.append(y_AL_temp)
    num_extra += y_AL_temp.shape[0]

for i in range(len(x_AL_list)):
    x_train_torch = torch.cat((x_train_torch, torch.from_numpy(x_AL_list[i]).float().reshape((x_AL_list[i].shape[0], 1, x_AL_list[i].shape[1]))), axis=0)
    y_train_torch = torch.cat((y_train_torch, torch.from_numpy(y_AL_list[i]).float()), axis=0)
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(x_train_torch.shape)
print(y_train_torch.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for param_group in optimizer.param_groups:
    param_group['lr'] = args.lr
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_loss_list = []
test_loss_list = []
early_stopping = EarlyStopping(max_num=100, min_delta=0.001)

for epoch in range(0, args.epochs):
    train(epoch, train_loss_list)
    test_loss = test(epoch, test_loss_list)
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

model = FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load('ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

l2_diff_1, sigma2_1, class_loss_1 = validation()

y_pred_torch_class, y_pred_torch_regression = model(x_study_torch.cuda())
y_pred_np = y_pred_torch_regression.detach().cpu().numpy().reshape(-1, 4)
w_reg = np.exp(y_pred_np[:,3])
w_reg = w_reg.astype(np.float64)
w_reg = w_reg / np.sum(w_reg)
prob = torch.nn.functional.softmax(y_pred_torch_class, dim=1)
entropy = -torch.sum(prob * torch.log(prob), dim=1).detach().cpu().numpy()
w_class = entropy / np.sum(entropy)
w = 0.5 * w_reg + 0.5 * w_class
freq = np.random.multinomial(args.ntrain, w)

with np.printoptions(threshold=np.inf):
    print("logits = ", y_pred_torch_class)
    print("prob = ", prob)
    print("entropy = ", entropy)
    print("logsig2 = ", y_pred_np[:,3])
    print("sig2 after norm = ", w_reg)
    print("entropy after norm = ", w_class)
    print("freq = ", freq)
    print("freq.shape = ", freq.shape)
    print("freq.sum = ", np.sum(freq))
#
np.save('AL-freq.npy', freq)

#----------------------baseline branch-------------------------
checkpoint = torch.load('ckpt_base.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

x_train = x_orig[0:args.ntrain+int(num_extra)]
y_train = y_orig[0:args.ntrain+int(num_extra)]
print(x_train.shape)
print(y_train.shape)

x_train_torch = torch.from_numpy(x_train).float()
x_train_torch = x_train_torch.reshape((x_train_torch.shape[0], 1, x_train_torch.shape[1]))
y_train_torch = torch.from_numpy(y_train).float()
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(x_train_torch.shape)
print(y_train_torch.shape)

for param_group in optimizer.param_groups:
    param_group['lr'] = args.lr
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_loss_list = []
test_loss_list = []
early_stopping = EarlyStopping(max_num=100, min_delta=0.001)

for epoch in range(0, args.epochs):
    train(epoch, train_loss_list)
    test_loss = test(epoch, test_loss_list)
    if epoch > args.blind_train_epoch:
        early_stopping(test_loss)
        if early_stopping.improve:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, 'ckpt_base.pth')
            print("Save model at epoch ", epoch)
        if early_stopping.do_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break;
print("Best val loss = ", early_stopping.best_loss)

model = FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load('ckpt_base.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

l2_diff_2, sigma2_2, class_loss_2 = validation()
