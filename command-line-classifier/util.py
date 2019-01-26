import os
import argparse
from torchvision import models
from torch import optim
from torch import nn
import torch
import PIL
import numpy as np
import json


img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])


def check_arch(arch):
    '''
    Checks for support of the desired architecture
    '''
    list_models = {'vgg13', 'vgg11', 'vgg19'}
    if not arch in list_models:
        raise argparse.ArgumentTypeError('{} is not in the supported architectures -> {}'.format(arch, list_models))
    return arch

def is_directory(directory):
    return os.path.isdir(directory) and os.path.exists(directory)

def check_directory(directory):
    '''Checks whether is it a directory and it exists
    '''
    if not is_directory(directory):
        raise argparse.ArgumentTypeError('{} is not a directory'.format(directory))
    return directory

def is_file(file):
    return os.path.isfile(file) and os.path.exists(file)

def check_file(file):
    '''Checks whether it is a file and it exists
    '''
    if not is_file(file):
        raise argparse.ArgumentTypeError('{} is not a file'.format(file))
    return file

def get_model(arch):
    '''
    Gets a model depending on the parameter
    '''
    model = None
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg11(pretrained=True)
    return model

def load_model(checkpoint_file):
    '''
    Loads a checkpoint file
    '''
    chkpt = torch.load(checkpoint_file)
    load_model = models.vgg11(pretrained=True)
    for params in load_model.parameters():
        params.requires_grad = False
    load_model.class_to_idx = chkpt['class_to_idx']
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(4096, 102),
                           nn.LogSoftmax(dim=1))
    load_model.classifier = classifier
    load_model.load_state_dict(chkpt['state_dict'])
    return load_model


def crop_image(image, crop_height, crop_width):
    '''Crop a PIL image
    '''
    left_margin = (image.width - crop_width)/2
    bottom_margin = (image.height - crop_height)/2
    right_margin = left_margin + crop_width
    top_margin = bottom_margin + crop_height
    return image.crop((left_margin, bottom_margin, right_margin, top_margin))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    if image.size[0] > image.size[1]:
        image.thumbnail((image.size[0], 256))
    else:
        image.thumbnail((256, image.size[1]))
    image = crop_image(image, 224, 224)
    image = np.array(image)/255
    image = (image-img_mean)/img_std
    image = image.transpose((2, 0, 1)) # PyTorch tensors assume the color channel is the first dimension
    return image

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    pil_im = PIL.Image.open(image_path, 'r')
    process_pil_im = torch.from_numpy(process_image(pil_im)).type(torch.FloatTensor)
    process_pil_im.unsqueeze_(0)
    process_pil_im = process_pil_im.to(device)
    test_pred = model(process_pil_im)
    ps = torch.exp(test_pred)
    top_pred, top_labels = ps.topk(topk)
    return top_pred.detach().cpu().numpy()[0], top_labels.detach().cpu().numpy()[0]

def show_flower_labels(probs, labels, category_names=None):
    ''' Display model probabilities and labels for a give image
    '''
    print("Top {} image category predictions".format(len(labels)))
    if category_names:
        for index, values in enumerate(zip(probs, labels)):
            print("{}.   Probability:{:.2f}%    Label Number:{}    Label:{}".format(index+1, values[0]*100, values[1]+1, category_names.get(str(values[1] + 1))))
    else:
        for index, values in enumerate(zip(probs, labels)):
            print("{}.   Probability: {:.2f}%    Label Number: {}".format(index+1, values[0]*100, values[1]+1))
            
def get_category_names(category_file):
    ''' Create a json object that contains label number to flower name
    '''
    if not is_file(category_file) or not category_file.endswith(".json"):
        raise parser.error("Make sure your category names file exist and has a .json extension")
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)