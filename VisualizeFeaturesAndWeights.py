import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from S3Net import s3net_new
import matplotlib.pyplot as plt

#%%
# load model, you shall need to change the directory name. This model is trained on GPU but fetched to CPU. So, map location is
# CPU and model.module is the desired output

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=172, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')



args = parser.parse_args()


def load(args):

    # create model


    model = s3net_new()
    

    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))   

    model = torch.nn.DataParallel(model)
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))


            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    
    return model.module


    
model_loaded = load(args)

model_loaded = model_loaded.to("cpu")

#%%
# Get an image to test 

import matplotlib.image as mpimg


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

test_dataset = datasets.ImageFolder(
        'C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]))

    
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1)
                       
images, labels = next(iter(test_loader))

print(images.shape)

#images = images.to("cpu")

plt.figure()

plt.imshow(images[0,1])
plt.show()    


#%%
# Get the output of each layer to visualize

activations = {}

def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook  

model_loaded.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation('conv2'))

from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(model_loaded,(3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))   
    
    
output = model_loaded(images)

act = activations['conv2'].squeeze()

layer1 = act.numpy()

model_loaded.features.denseblock1.denselayer2.conv2.register_forward_hook(get_activation('conv2'))

output = model_loaded(images)

act = activations['conv2'].squeeze()

layer2 = act.numpy()

model_loaded.features.denseblock1.denselayer3.conv2.register_forward_hook(get_activation('conv2'))

output = model_loaded(images)

act = activations['conv2'].squeeze()

layer3 = act.numpy()

model_loaded.features.denseblock1.denselayer4.conv2.register_forward_hook(get_activation('conv2'))

output = model_loaded(images)

act = activations['conv2'].squeeze()

layer4 = act.numpy()


model_loaded.features.denseblock1.denselayer5.conv2.register_forward_hook(get_activation('conv2'))

output = model_loaded(images)

act = activations['conv2'].squeeze()

layer5 = act.numpy()

model_loaded.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation('conv2'))

output = model_loaded(images)

act = activations['conv2'].squeeze()

layer6 = act.numpy()

#%%
# Get the weights of the final layer

kernels = model_loaded.features.transition1.conv.weight.data.clone()

kernels = kernels.squeeze()

kernels = kernels.numpy()[:,32:992]

I1 = np.sum(np.abs(kernels[:,0:160]))
I2 = np.sum(np.abs(kernels[:,160:320]))
I3 = np.sum(np.abs(kernels[:,320:480]))
I4 = np.sum(np.abs(kernels[:,480:640]))
I5 = np.sum(np.abs(kernels[:,640:800]))
I6 = np.sum(np.abs(kernels[:,800:960]))


#%%
# Visualize the features produced by each layer in the first block


import matplotlib.pyplot as plt
import matplotlib.image as im
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

csfont = {'fontname':'Times New Roman'}

plt.figure(dpi = 280)
gs1 = gridspec.GridSpec(6, 8)
gs1.update(wspace=0.025, hspace=0.09) # set the spacing between axes. 

ax=plt.subplot(gs1[0])
fig=plt.imshow(layer1[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[1])
fig=plt.imshow(layer1[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[2])
fig=plt.imshow(layer1[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[3])
fig=plt.imshow(layer1[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[4])
fig=plt.imshow(layer1[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[5])
fig=plt.imshow(layer1[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[6])
fig=plt.imshow(layer1[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[7])
fig=plt.imshow(layer1[45],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-1', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_title("Clean Image")



ax=plt.subplot(gs1[8])
fig=plt.imshow(layer2[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[9])
fig=plt.imshow(layer2[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[10])
fig=plt.imshow(layer2[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[11])
fig=plt.imshow(layer2[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[12])
fig=plt.imshow(layer2[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[13])
fig=plt.imshow(layer2[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[14])
fig=plt.imshow(layer2[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[15])
fig=plt.imshow(layer2[45],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-2', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_title("Clean Image")

    
    
ax=plt.subplot(gs1[16])
fig=plt.imshow(layer3[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[17])
fig=plt.imshow(layer3[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[18])
fig=plt.imshow(layer3[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[19])
fig=plt.imshow(layer3[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[20])
fig=plt.imshow(layer3[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[21])
fig=plt.imshow(layer3[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[22])
fig=plt.imshow(layer3[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[23])
fig=plt.imshow(layer3[45],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-3', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_title("Clean Image")

    
    
ax=plt.subplot(gs1[24])
fig=plt.imshow(layer4[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[25])
fig=plt.imshow(layer4[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[26])
fig=plt.imshow(layer4[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[27])
fig=plt.imshow(layer4[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[28])
fig=plt.imshow(layer4[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[29])
fig=plt.imshow(layer4[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[30])
fig=plt.imshow(layer4[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[31])
fig=plt.imshow(layer4[45],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-4', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([]) 



ax=plt.subplot(gs1[32])
fig=plt.imshow(layer5[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[33])
fig=plt.imshow(layer5[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[34])
fig=plt.imshow(layer5[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[35])
fig=plt.imshow(layer5[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[36])
fig=plt.imshow(layer5[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[37])
fig=plt.imshow(layer5[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[38])
fig=plt.imshow(layer5[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[39])
fig=plt.imshow(layer5[45],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-5', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([]) 



ax=plt.subplot(gs1[40])
fig=plt.imshow(layer6[0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[41])
fig=plt.imshow(layer6[10], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[42])
fig=plt.imshow(layer6[32],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[43])
fig=plt.imshow(layer6[55], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[44])
fig=plt.imshow(layer6[5],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[45])
fig=plt.imshow(layer6[51],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[46])
fig=plt.imshow(layer6[25],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[47])
fig=plt.imshow(layer6[45],origin="upper", cmap='gray')
#plt.title('Cov-Layer-6',**csfont,fontsize=8)

plt.text(0.5, 0.9,'Cov-Layer-6', **csfont,fontsize=5, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([]) 








   
    
