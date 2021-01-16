import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import time, copy, sys, os


from PIL import Image

from se_resnet import se_resnet50

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            #if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(fname,['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

class TImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, y) where y is the label of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

    def __len__(self):
        return len(self.imgs)

data_dir = 'tiny-imagenet-200/'

num_workers = {'train' : 10,'val'   : 0,'test'  : 0}

traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'val')
train_batch = 96
test_batch = 200


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch, shuffle=True,
        num_workers=num_workers['train'], pin_memory=True)
    
valdir = os.path.join(data_dir, 'val', 'images')
valgtfile = os.path.join(data_dir, 'val', 'val_annotations.txt')
val_dataset = TImgNetDataset(valdir, valgtfile, class_to_idx=train_loader.dataset.class_to_idx.copy(),
            transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
            ]))
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch, shuffle=False,
        num_workers=num_workers['val'], pin_memory=True)



dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

dataloaders = {'train': train_loader, 'val': val_loader}

model_ft = se_resnet50(num_classes=1000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
print(sum(p.numel() for p in model_ft.parameters() if p.requires_grad))
#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
Init_lr = 0.005
optimizer_ft = optim.SGD(model_ft.parameters(), lr=Init_lr, momentum=0.9)


def adjust_learning_rate(optimizer, epoch, Init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = Init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(param_group['lr'])



def train_model(output_path, model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=100, scheduler=None):
    if not os.path.exists('models/'+str(output_path)):
        os.makedirs('models/'+str(output_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        adjust_learning_rate(optimizer_ft, epoch, Init_lr)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")

#                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        torch.save(model.state_dict(), './models/' + str(output_path) + '/model_{}_epoch.pt'.format(epoch+1))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))
    

def test_model(model, dataloaders, dataset_sizes, criterion, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['test']:
        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i,(inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
            sys.stdout.flush()


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print()
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print()
    
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))   

    
train_model("fenglei/Alex",model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=90)

test_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft)


