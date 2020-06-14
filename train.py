import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

import json
import argparse
import os
import sys

def args_parser():
    parser = argparse.ArgumentParser(description = "Training file")
    
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Add your dataset directory.')
    parser.add_argument('--gpu', type = bool, default = 'True', help = 'True: train with gpu, False: with cpu')
    parser.add_argument('--epochs', type = int, default = 10, help = 'total number of training episode')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'model architecture, default is vgg 16, densenet121 is also available')
    parser.add_argument('--hidden_units', type = int, default = 500, help = 'number of hidden unit')
    parser.add_argument('--save_dir', type = str, default = 'p2checkpoint.pth', help = 'save the check point')
    args = parser.parse_args()
    return args

def process_data(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(40),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                          ])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(60),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                          ])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                          ])
    
    train_set = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_set = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 64, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
    
    return train_loader, valid_loader, test_loader, train_set, valid_set, test_set

def default_model(arch):
    if arch == 'vgg16':
        load_model = models.vgg16(pretrained = True)
        print("You are using default vgg 16 model.")
    elif arch == "densenet":
        load_model = models.densenet121(pretrained = True)
        print("You are using densenet 121 model.")
    else:
        print("There is only two model to choose. performing the test with default model bgg 16")
        load_model = models.vgg16(pretrained = True)
        
    return load_model
    
def set_classifier(load_model, hidden_units):
    if hidden_units == None:
        hidden_units = 512
    inputs = load_model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(inputs, hidden_units, bias = True)),
                                           ('relu1', nn.ReLU()),
                                           ('dropout', nn.Dropout(p = 0.4)),
                                           ('fc2', nn.Linear(hidden_units, 128, bias = True)),
                                           ('relu2', nn.ReLU()),
                                           ('dropout', nn.Dropout(p = 0.4)),
                                           ('fc3', nn.Linear(128, 102, bias = True)),
                                           ('output', nn.LogSoftmax(dim = 1))
                                           ]))
    
    return classifier
    
def train_model(epochs, train_loader, valid_loader, device, model, optimizer, criterion):
    if type(epochs) == type(None):
        epochs = 10
        print('Epochs = 10')
        
    steps = 0
    model.to(device)
    running_loss = 0
    print_every = 100
    
    for epoch in range(epochs):
        for img, labels in train_loader:
            steps += 1
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(img)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps%print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for img, labels in valid_loader:
                        img, labels = img.to(device), labels.to(device)
                        logps = model.forward(img)
                        loss = criterion(logps, labels)
                        
                        test_loss += loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print('Epoch : {}/{}.....Training loss : {:.3f}... Valid loss : {:.3f}.... valid accuracy : {:.3f}'.format(epoch +1, epochs,running_loss/print_every, test_loss/len(valid_loader), accuracy/len(valid_loader)))
                running_loss = 0
                model.train()
    return model

def test_model(model, test_loader, device, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for img, labels in test_loader:
            img, labels = img.to(device), labels.to(device)
            logps = model.forward(img)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print('Test loss : {:.3f} ........ Test accuracy : {:.3f}'.format(test_loss/len(test_loader), accuracy/len(test_loader)))        
    
def save_checkpoint(model, train_set, save_dir, arch):
    model.class_to_idx = train_set.class_to_idx
    checkpoint = {'structure': arch,
                 'classifier': model.classifier,
                 'state_dic': model.state_dict(),
                 'class_to_idx': model.class_to_idx}
    return torch.save(checkpoint, save_dir)

def main():
    args = args_parser()
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cpu')
    if is_gpu and use_cuda:
        device = torch.device('cuda:0')
        print('Device is set to {}'.format(device))
    else:
        device = torch.device('cpu')
        print('Device is set to {}'.format(device))
        
    data_dir = 'flowers'
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    
    train_loader, valid_loader, test_loader, train_set, valid_set, test_set = process_data(train_dir, valid_dir, test_dir)
    
    model = default_model(args.arch)
    
    for params in model.parameters():
        params.requires_grad = False
        
    model.classifier = set_classifier(model, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    tr_model = train_model(args.epochs, train_loader, valid_loader, device, model, optimizer, criterion)
    test_model(tr_model, test_loader, device, criterion)
    save_checkpoint(tr_model, train_set, args.save_dir, args.arch)
    print('All finished')
    
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    
    
                        
                        
    
    
    
    
    
    
    