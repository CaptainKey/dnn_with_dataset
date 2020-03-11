
import datetime
import os
import argparse
import logging as log
from inspect import getmembers, isclass
import networks
from tqdm import tqdm
def prepare_env():
    time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    directory = 'export/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory += time
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def parse_arg():
    parser = argparse.ArgumentParser(description='Neural Network with dataset')
    parser.add_argument('--epochs', metavar='int', type=int, nargs='?',required=False,default=1,help='nb epochs for training')
    parser.add_argument('--train', metavar='int', type=int, nargs='?',required=False,default=1,help='1 for training 0 for testing')
    parser.add_argument('--checkpoint', metavar='str', type=str, nargs='?',required=False,default=False,help='path to checkpoint file')
    parser.add_argument('--batch', metavar='int', type=int, nargs='?',required=False,default=4,help='batch argument for training and testing')
    parser.add_argument('--arch', metavar='str', type=str, nargs='?',required=False,default=False,help='name of network architecture to use')

    args = parser.parse_args()

    return args.train,args.checkpoint,args.epochs,args.batch,args.arch


def choose_network(arch):
    networks_definition = [net for net in getmembers(networks) if isclass(net[1])]
    if arch:
        try:
            network = [net for net in networks_definition if net[0] == arch][0]
        except IndexError:
            log.error('Could not find network : {}'.format(arch))
            exit('Could not find network : {}'.format(arch))
    else:
        tqdm.write('Choose a network : ')

        for i,nets in enumerate(networks_definition):
            name, net = nets
            tqdm.write('\t {} - {}'.format(i,name))

        choice = int(input('Choose your net : '))

        name = network = networks_definition[choice][0]
        network = networks_definition[choice][1]()

    return name,network