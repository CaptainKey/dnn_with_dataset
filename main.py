import torch
import torch.nn as nn 
import torch.optim as optim

import torchvision.transforms as transforms

from dataset import dataset
import logging as log

from utils import *
from compute import *


export_directory = prepare_env()

log.basicConfig(filename='{}/debug.log'.format(export_directory),format='%(levelname)s : %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=log.INFO,filemode='w')

train, checkpoint,epochs,batch,arch = parse_arg()

if not arch:
    name, network = choose_network(arch)
    log.info('Network : {} loaded ! '.format(name))


if checkpoint:
    log.info('Loading chekpoint : {}'.format(checkpoint))
    network.load_state_dict(torch.load(checkpoint))


transform = transforms.Compose([transforms.ToTensor()])


testset = dataset('test set',transform,train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)

if train:
    trainset = dataset('train set',transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,  shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    network = training(network,criterion,optimizer,trainloader,epochs)
    

performance = test(network,testloader)
log.info('Saving model : {}/checkpoint.{}.pt'.format(export_directory,performance))
torch.save(network.state_dict(), '{}/checkpoint{}.pt'.format(export_directory,performance))







