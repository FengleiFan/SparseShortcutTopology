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
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import torchvision
import torchvision.transforms as T


from torchsummary import summary

from PIL import Image

from full_grad import FullGrad

#%%
# load the proposed network
# load model, you shall need to change the directory name. This model is trained on GPU but fetched to CPU. So, map location is
# CPU and model.module is the desired output

resume_dir = './large/model_best.pth.tar'

def load(resume_dir):

    # create model


    model = s3net_new()
    

    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))   

    model = torch.nn.DataParallel(model)
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    # optionally resume from a checkpoint
    if resume_dir:
        if os.path.isfile(resume_dir):
            print("=> loading checkpoint '{}'".format(resume_dir))


            checkpoint = torch.load(resume_dir, map_location="cpu")

            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_dir, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_dir))
    
    
    return model.module



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


#

def compute_dice(seg_map, saliency, percentile):

    percentile_number = np.percentile(saliency, percentile)
    
    saliency[saliency < percentile_number] = 0
    
    #saliency[saliency >= percentile_number] = 1
    saliency[saliency > 0] = 1
      
    dice = np.sum(saliency[seg_map==1])*2.0 / (np.sum(saliency) + np.sum(seg_map))

    return dice  

def get_saliency_map(X, model_loaded):
    
    fullgrad_proposed = FullGrad(model_loaded)
    
    fullgrad_proposed.checkCompleteness()
    
    
    # Obtain saliency maps
    saliency_proposed = fullgrad_proposed.saliency(X)
    
    saliency_numpy_proposed = saliency_proposed.detach().numpy()
    saliency_numpy_proposed[saliency_numpy_proposed<0] = 0
    
    saliency_proposed = saliency_numpy_proposed[0,0]

    return  saliency_proposed    


def get_dice_for_model(X, seg_map, model_name, resume_dir = resume_dir):
    
    
    if model_name == 'proposed':
           model_loaded = load(resume_dir)   
    
    if model_name == 'vgg19':
           model_loaded = torchvision.models.vgg19(pretrained=True)
           
    if model_name == 'squeezenet':
           model_loaded = torchvision.models.squeezenet1_0(pretrained=True)           

    if model_name == 'resnet50':
           model_loaded = torchvision.models.resnet50(pretrained=True)  
           

    if model_name == 'densenet169':
           model_loaded = torchvision.models.densenet169(pretrained=True)  
    
    for param in model_loaded.parameters():
           param.requires_grad = False

    saliency_map = get_saliency_map(X, model_loaded) 
    
    print(saliency_map.shape)
    dice = np.zeros((1,20))
    map_get = np.zeros((224, 224))

    for i in np.arange(20):
    
           map_get[:,:] = saliency_map
           dice[0,i] = compute_dice(seg_map, map_get, 99-i)

    return dice
    

def get_segmentation_for_model(X, model_name, resume_dir = resume_dir):
    
    
    if model_name == 'proposed':
           model_loaded = load(resume_dir)   
    
    if model_name == 'vgg19':
           model_loaded = torchvision.models.vgg19(pretrained=True)
           
    if model_name == 'squeezenet':
           model_loaded = torchvision.models.squeezenet1_0(pretrained=True)           

    if model_name == 'resnet50':
           model_loaded = torchvision.models.resnet50(pretrained=True)  
           

    if model_name == 'densenet169':
           model_loaded = torchvision.models.densenet169(pretrained=True)  
    
    for param in model_loaded.parameters():
           param.requires_grad = False

    saliency = get_saliency_map(X, model_loaded) 
    
    percentile_number = np.percentile(saliency, 90)
    
    saliency[saliency < percentile_number] = 0
    
    #saliency[saliency >= percentile_number] = 1
    saliency[saliency > 0] = 1
    
    return saliency
    
#%%

img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/common_newt.jpeg') 

seg = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/seg_common_newt.png').convert('RGB') 

X = preprocess(img)  

X_seg = preprocess(seg)  

seg_map = deprocess(X_seg)

seg_map = seg_map.detach().numpy()
seg_map[seg_map<0] = 0
seg_map[seg_map>0] = 1

seg_map_common_newt = seg_map[0]




common_newt_seg_proposed = get_segmentation_for_model(X, 'proposed')

common_newt_seg_vgg19 = get_segmentation_for_model(X, 'vgg19')

common_newt_seg_squeezenet = get_segmentation_for_model(X, 'squeezenet')

common_newt_seg_resnet50 = get_segmentation_for_model(X, 'resnet50')

common_newt_seg_densenet169 = get_segmentation_for_model(X, 'densenet169')


#%%
img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/brambling.jpeg') 

seg = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/seg_brambling.png').convert('RGB') 

X = preprocess(img)  

X_seg = preprocess(seg)  

seg_map = deprocess(X_seg)

seg_map = seg_map.detach().numpy()
seg_map[seg_map<0] = 0
seg_map[seg_map>0] = 1

seg_map_brambling = seg_map[0]



brambling_seg_proposed = get_segmentation_for_model(X, 'proposed')

brambling_seg_vgg19 = get_segmentation_for_model(X, 'vgg19')

brambling_seg_squeezenet = get_segmentation_for_model(X, 'squeezenet')

brambling_seg_resnet50 = get_segmentation_for_model(X, 'resnet50')

brambling_seg_densenet169 = get_segmentation_for_model(X, 'densenet169')




#%%

img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/quail.jpeg') 

seg = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/seg_quail.png').convert('RGB') 

X = preprocess(img)  

X_seg = preprocess(seg)  

seg_map = deprocess(X_seg)

seg_map = seg_map.detach().numpy()
seg_map[seg_map<0] = 0
seg_map[seg_map>0] = 1

seg_map_quail = seg_map[0]



quail_seg_proposed = get_segmentation_for_model(X, 'proposed')

quail_seg_vgg19 = get_segmentation_for_model(X, 'vgg19')

quail_seg_squeezenet = get_segmentation_for_model(X, 'squeezenet')

quail_seg_resnet50 = get_segmentation_for_model(X, 'resnet50')

quail_seg_densenet169 = get_segmentation_for_model(X, 'densenet169')




#%%


img = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/val/corn.jpeg') 

seg = Image.open('C:/Users/Fenglei Fan/Desktop/PhD document/S3Net/ReviseBasedOnTNNLS/interpretability/seg_corn.png').convert('RGB') 

X = preprocess(img)  

X_seg = preprocess(seg)  

seg_map = deprocess(X_seg)

seg_map = seg_map.detach().numpy()
seg_map[seg_map<0] = 0
seg_map[seg_map>0] = 1

seg_map_corn = seg_map[0]



corn_seg_proposed = get_segmentation_for_model(X, 'proposed')

corn_seg_vgg19 = get_segmentation_for_model(X, 'vgg19')

corn_seg_squeezenet = get_segmentation_for_model(X, 'squeezenet')

corn_seg_resnet50 = get_segmentation_for_model(X, 'resnet50')

corn_seg_densenet169 = get_segmentation_for_model(X, 'densenet169')



#%%
csfont = {'fontname':'Times New Roman'}

figfig = plt.figure(dpi = 280)

gs1 = gridspec.GridSpec(4, 6)
gs1.update(wspace=0.025, hspace=0.09) # set the spacing between axes. 

ax=plt.subplot(gs1[0])
fig=plt.imshow(seg_map_common_newt, origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])



ax=plt.subplot(gs1[1])
fig=plt.imshow(common_newt_seg_vgg19,origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)



ax=plt.subplot(gs1[2])
fig=plt.imshow(common_newt_seg_squeezenet, origin="upper", cmap='gray')
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
fig=plt.imshow(common_newt_seg_resnet50,origin="upper", cmap='gray')
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
fig=plt.imshow(common_newt_seg_densenet169,origin="upper", cmap='gray')
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
fig=plt.imshow(common_newt_seg_proposed, origin="upper", cmap='gray')
plt.text(0.5, 0.9,'Proposed', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[6])
fig=plt.imshow(seg_map_brambling, origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[7])
fig=plt.imshow(brambling_seg_vgg19, origin="upper", cmap='gray')
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
fig=plt.imshow(brambling_seg_squeezenet,origin="upper", cmap='gray')
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
fig=plt.imshow(brambling_seg_resnet50, origin="upper", cmap='gray')
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
fig=plt.imshow(brambling_seg_densenet169,origin="upper", cmap='gray')
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
fig=plt.imshow(brambling_seg_proposed,origin="upper", cmap='gray')
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
fig=plt.imshow(seg_map_quail, origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[13])
fig=plt.imshow(quail_seg_vgg19, origin="upper", cmap='gray')
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
fig=plt.imshow(quail_seg_squeezenet,origin="upper", cmap='gray')
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
fig=plt.imshow(quail_seg_resnet50, origin="upper", cmap='gray')
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
fig=plt.imshow(quail_seg_densenet169,origin="upper", cmap='gray')
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
fig=plt.imshow(quail_seg_proposed,origin="upper", cmap='gray')
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
fig=plt.imshow(seg_map_corn,origin="upper", cmap='gray')

plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


   
ax=plt.subplot(gs1[19])
fig=plt.imshow(corn_seg_vgg19, origin="upper", cmap='gray')
plt.text(0.5, 0.9,'VGG19', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[20])
fig=plt.imshow(corn_seg_squeezenet, origin="upper", cmap='gray')
plt.text(0.5, 0.9,'SqueezeNet', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#ax.set_title("Leaky-Res-AE")

ax=plt.subplot(gs1[21])
fig=plt.imshow(corn_seg_resnet50,origin="upper", cmap='gray')
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
fig=plt.imshow(corn_seg_densenet169, origin="upper", cmap='gray')
plt.text(0.5, 0.9,'DeneseNet169', **csfont,fontsize=6, color = 'r',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()

ax.set_xticklabels([])
ax.set_yticklabels([])
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)




ax=plt.subplot(gs1[23])
fig=plt.imshow(corn_seg_proposed,origin="upper", cmap='gray')
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
