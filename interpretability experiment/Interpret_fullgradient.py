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

import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from S3Net import s3net_large
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torchvision
import torchvision.transforms as T


from torchsummary import summary

from PIL import Image

#%%
# load the proposed network
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
parser.add_argument('--resume', default='./large/model_best.pth.tar', type=str, metavar='PATH',
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

#%%

def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])
    return transform(image)




#%%
from full_grad import FullGrad

# Get an image to test 

import matplotlib.image as mpimg


# Opening the image

#fish
#img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/fish.jpeg') 


#hen
img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/common_newt.jpeg') 



X = preprocess(img)  

Y = deprocess(X)


#%%


X_numpy = Y.detach().numpy()

                 
X_numpy_common_newt = X_numpy.transpose(1,2,0)

model_loaded = load(args)

model_loaded = model_loaded.to("cpu")
model_loaded.eval()

for param in model_loaded.parameters():
    param.requires_grad = False


fullgrad_proposed = FullGrad(model_loaded)

fullgrad_proposed.checkCompleteness()


# Obtain saliency maps
saliency_proposed_common_newt = fullgrad_proposed.saliency(X)

saliency_numpy_proposed_common_newt = saliency_proposed_common_newt.detach().numpy()


model_loaded = torchvision.models.vgg19(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_vgg19 = FullGrad(model_loaded)

fullgrad_vgg19.checkCompleteness()


# Obtain saliency maps
saliency_vgg19_common_newt = fullgrad_vgg19.saliency(X)

saliency_numpy_vgg19_common_newt = saliency_vgg19_common_newt.detach().numpy()



model_loaded = models.squeezenet1_0(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_SqueezeNet = FullGrad(model_loaded)

fullgrad_SqueezeNet.checkCompleteness()


# Obtain saliency maps
saliency_SqueezeNet_common_newt = fullgrad_SqueezeNet.saliency(X)

saliency_numpy_SqueezeNet_common_newt = saliency_SqueezeNet_common_newt.detach().numpy()



model_loaded = models.resnet50(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_resnet50 = FullGrad(model_loaded)

fullgrad_resnet50.checkCompleteness()


# Obtain saliency maps
saliency_resnet50_common_newt = fullgrad_resnet50.saliency(X)
saliency_numpy_resnet50_common_newt = saliency_resnet50_common_newt.detach().numpy()



model_loaded = models.densenet169(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_densenet169 = FullGrad(model_loaded)

fullgrad_densenet169.checkCompleteness()


# Obtain saliency maps
saliency_densenet169_common_newt = fullgrad_densenet169.saliency(X)

saliency_numpy_densenet169_common_newt = saliency_densenet169_common_newt.detach().numpy()




img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/brambling.jpeg') 




X = preprocess(img)  

Y = deprocess(X)


X_numpy = Y.detach().numpy()

                 
X_numpy_brambling = X_numpy.transpose(1,2,0)

model_loaded = load(args)

model_loaded = model_loaded.to("cpu")
model_loaded.eval()

for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_proposed = FullGrad(model_loaded)

fullgrad_proposed.checkCompleteness()


# Obtain saliency maps
saliency_proposed_brambling = fullgrad_proposed.saliency(X)

saliency_numpy_proposed_brambling = saliency_proposed_brambling.detach().numpy()


model_loaded = torchvision.models.vgg19(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_vgg19 = FullGrad(model_loaded)

fullgrad_vgg19.checkCompleteness()


# Obtain saliency maps
saliency_vgg19_brambling = fullgrad_vgg19.saliency(X)

saliency_numpy_vgg19_brambling = saliency_vgg19_brambling.detach().numpy()



model_loaded = models.squeezenet1_0(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_SqueezeNet = FullGrad(model_loaded)

fullgrad_SqueezeNet.checkCompleteness()


# Obtain saliency maps
saliency_SqueezeNet_brambling = fullgrad_SqueezeNet.saliency(X)


saliency_numpy_SqueezeNet_brambling = saliency_SqueezeNet_brambling.detach().numpy()


model_loaded = models.resnet50(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_resnet50 = FullGrad(model_loaded)

fullgrad_resnet50.checkCompleteness()


# Obtain saliency maps
saliency_resnet50_brambling = fullgrad_resnet50.saliency(X)

saliency_numpy_resnet50_brambling = saliency_resnet50_brambling.detach().numpy()



model_loaded = models.densenet169(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_densenet169 = FullGrad(model_loaded)

fullgrad_densenet169.checkCompleteness()


# Obtain saliency maps
saliency_densenet169_brambling = fullgrad_densenet169.saliency(X)

saliency_numpy_densenet169_brambling = saliency_densenet169_brambling.detach().numpy()











img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/quail.jpeg') 


X = preprocess(img)  

Y = deprocess(X)



X_numpy = Y.detach().numpy()

                 
X_numpy_quail = X_numpy.transpose(1,2,0)

model_loaded = load(args)

model_loaded = model_loaded.to("cpu")
model_loaded.eval()

for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_proposed = FullGrad(model_loaded)

fullgrad_proposed.checkCompleteness()


# Obtain saliency maps
saliency_proposed_quail = fullgrad_proposed.saliency(X)

saliency_numpy_proposed_quail = saliency_proposed_quail.detach().numpy()


model_loaded = torchvision.models.vgg19(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_vgg19 = FullGrad(model_loaded)

fullgrad_vgg19.checkCompleteness()


# Obtain saliency maps
saliency_vgg19_quail = fullgrad_vgg19.saliency(X)

saliency_numpy_vgg19_quail = saliency_vgg19_quail.detach().numpy()



model_loaded = models.squeezenet1_0(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_SqueezeNet = FullGrad(model_loaded)

fullgrad_SqueezeNet.checkCompleteness()


# Obtain saliency maps
saliency_SqueezeNet_quail = fullgrad_SqueezeNet.saliency(X)


saliency_numpy_SqueezeNet_quail = saliency_SqueezeNet_quail.detach().numpy()

model_loaded = models.resnet50(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_resnet50 = FullGrad(model_loaded)

fullgrad_resnet50.checkCompleteness()


# Obtain saliency maps
saliency_resnet50_quail = fullgrad_resnet50.saliency(X)

saliency_numpy_resnet50_quail = saliency_resnet50_quail.detach().numpy()



model_loaded = models.densenet169(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_densenet169 = FullGrad(model_loaded)

fullgrad_densenet169.checkCompleteness()


# Obtain saliency maps
saliency_densenet169_quail = fullgrad_densenet169.saliency(X)

saliency_numpy_densenet169_quail = saliency_densenet169_quail.detach().numpy()





img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/corn.jpeg') 

X = preprocess(img)  

Y = deprocess(X)



X_numpy = Y.detach().numpy()

                 
X_numpy_corn = X_numpy.transpose(1,2,0)

model_loaded = load(args)

model_loaded = model_loaded.to("cpu")
model_loaded.eval()

for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_proposed = FullGrad(model_loaded)

fullgrad_proposed.checkCompleteness()


# Obtain saliency maps
saliency_proposed_corn = fullgrad_proposed.saliency(X)

saliency_numpy_proposed_corn = saliency_proposed_corn.detach().numpy()


model_loaded = torchvision.models.vgg19(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_vgg19 = FullGrad(model_loaded)

fullgrad_vgg19.checkCompleteness()


# Obtain saliency maps
saliency_vgg19_corn = fullgrad_vgg19.saliency(X)

saliency_numpy_vgg19_corn = saliency_vgg19_corn.detach().numpy()



model_loaded = models.squeezenet1_0(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_SqueezeNet = FullGrad(model_loaded)

fullgrad_SqueezeNet.checkCompleteness()


# Obtain saliency maps
saliency_SqueezeNet_corn = fullgrad_SqueezeNet.saliency(X)


saliency_numpy_SqueezeNet_corn = saliency_SqueezeNet_corn.detach().numpy()


model_loaded = models.resnet50(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_resnet50 = FullGrad(model_loaded)

fullgrad_resnet50.checkCompleteness()


# Obtain saliency maps
saliency_resnet50_corn = fullgrad_resnet50.saliency(X)

saliency_numpy_resnet50_corn = saliency_resnet50_corn.detach().numpy()



model_loaded = models.densenet169(pretrained=True)
model_loaded.eval()
for param in model_loaded.parameters():
    param.requires_grad = False

fullgrad_densenet169 = FullGrad(model_loaded)

fullgrad_densenet169.checkCompleteness()


# Obtain saliency maps
saliency_densenet169_corn = fullgrad_densenet169.saliency(X)

saliency_numpy_densenet169_corn = saliency_densenet169_corn.detach().numpy()



#%%


csfont = {'fontname':'Times New Roman'}

figfig = plt.figure(dpi = 280)

gs1 = gridspec.GridSpec(4, 6)
gs1.update(wspace=0.025, hspace=0.09) # set the spacing between axes. 

ax=plt.subplot(gs1[0])
fig=plt.imshow(X_numpy_common_newt, origin="upper", cmap=plt.cm.hot)

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[1])
fig=plt.imshow(saliency_numpy_vgg19_common_newt[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
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
fig=plt.imshow(saliency_numpy_SqueezeNet_common_newt[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'SqueezeNet', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[3])
fig=plt.imshow(saliency_numpy_resnet50_common_newt[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'ResNet50', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[4])
fig=plt.imshow(saliency_numpy_densenet169_common_newt[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'DenseNet169', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)





ax=plt.subplot(gs1[5])
fig=plt.imshow(saliency_numpy_proposed_common_newt[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Proposed', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[6])
fig=plt.imshow(X_numpy_brambling, origin="upper", cmap=plt.cm.hot)

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[7])
fig=plt.imshow(saliency_numpy_vgg19_brambling[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[8])
fig=plt.imshow(saliency_numpy_SqueezeNet_brambling[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'SqueezeNet', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[9])
fig=plt.imshow(saliency_numpy_resnet50_brambling[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'ResNet50', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[10])
fig=plt.imshow(saliency_numpy_densenet169_brambling[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'DenseNet169', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)





ax=plt.subplot(gs1[11])
fig=plt.imshow(saliency_numpy_proposed_brambling[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Proposed', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


ax=plt.subplot(gs1[12])
fig=plt.imshow(X_numpy_quail, origin="upper", cmap=plt.cm.hot)

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[13])
fig=plt.imshow(saliency_numpy_vgg19_quail[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[14])
fig=plt.imshow(saliency_numpy_SqueezeNet_quail[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'SqueezeNet', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[15])
fig=plt.imshow(saliency_numpy_resnet50_quail[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'ResNet50', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[16])
fig=plt.imshow(saliency_numpy_densenet169_quail[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'DenseNet169', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)





ax=plt.subplot(gs1[17])
fig=plt.imshow(saliency_numpy_proposed_quail[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Proposed', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


   
ax=plt.subplot(gs1[18])
fig=plt.imshow(X_numpy_corn, origin="upper", cmap=plt.cm.hot)

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[19])
fig=plt.imshow(saliency_numpy_vgg19_corn[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[20])
fig=plt.imshow(saliency_numpy_SqueezeNet_corn[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'SqueezeNet', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[21])
fig=plt.imshow(saliency_numpy_resnet50_corn[0,0], origin="upper", cmap='gray')
plt.text(0.5, 0.9,'ResNet50', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

ax=plt.subplot(gs1[22])
fig=plt.imshow(saliency_numpy_densenet169_corn[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'DenseNet169', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[23])
fig=plt.imshow(saliency_numpy_proposed_corn[0,0],origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Proposed', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)       
    
#figfig.suptitle('FullGrad', **csfont, size=16)