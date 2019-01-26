import argparse
import PIL
import torch
import time
from torch import optim
from torch import nn
import numpy as np
import torch.nn.functional as F
import workspace_utils as wpu
from torchvision import datasets, transforms, models
import util


parser = argparse.ArgumentParser(description='Train a new network on a data set')

parser.add_argument('data_path', nargs='?', type=util.check_directory)
parser.add_argument('--save_dir', action='store', help='File name for saved model checkpoint')
parser.add_argument('--arch', action="store", default='vgg11', type=util.check_arch, help='Name of pretrained pytorch vision model')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float, help='Value for your learning rate')
parser.add_argument('--hidden_units', action='store', default=4096, type=int, help='Number of hidden_units')
parser.add_argument('--epochs', action='store', default=3, type=int, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', default=False, help='Training via gpu')

args = parser.parse_args()

if args.data_path == None:
    parser.error('Data path was not specified')
    
epochs = args.epochs
hidden_units = args.hidden_units
chkpt_file = args.save_dir
learning_rate = args.learning_rate
arch_name = args.arch
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
data_dir = args.data_path
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

if not util.is_directory(train_dir):
    raise Exception('{} is not a directory'.format(train_dir))

if not util.is_directory(valid_dir):
    raise Exception('{} is not a directory'.format(train_dir))

    
print('Building a {} model using a {} learning_rate, {} hidden_units, \
{} epochs and training using a {} device'.format(arch_name, learning_rate, hidden_units, epochs, device))

transforms = { 'train': transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(util.img_mean, util.img_std)]),
               'test': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(util.img_mean, util.img_std)]) 
             }

datasets = {'train': datasets.ImageFolder(train_dir, transform=transforms['train']),
            'valid': datasets.ImageFolder(valid_dir, transform=transforms['test'])
           }

imageloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True),
                'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=64, shuffle=True) 
               }

model = util.get_model(arch_name)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))
optimizer = None
model.classifier = classifier
optimizier = optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
model.to(device)

running_loss = 0
step = 0
print("Started training and validation")
with wpu.active_session():
    for e in range(epochs):
        start = time.time()
        for images, labels in imageloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizier.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizier.step()
            running_loss += loss.item()
        else:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in imageloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    val_loss += loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print("Epoch {}/{}.. ".format(e+1, args.epochs),
                  "Training loss: {:.3f}.. ".format(running_loss/len(imageloaders['train'])),
                  "Validation loss: {:.3f}.. ".format(val_loss/len(imageloaders['valid'])),
                  "Accuracy: {:.3f}.. ".format(accuracy/len(imageloaders['valid'])),
                  "Time to train: {:.3f}.. ".format((time.time() - start)/60))
            running_loss = 0
            model.train()
            
print("Finished training and validation")

if not chkpt_file == None:
    model.class_to_idx = datasets['train'].class_to_idx
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()
                 }
    checkpoint_file =  chkpt_file if '.pth' in chkpt_file else chkpt_file + '.pth'
    torch.save(checkpoint, checkpoint_file)
    print('Finished saving model to {}'.format(checkpoint_file))
