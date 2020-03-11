import torch
import logging as log
from tqdm import tqdm

def training(network,criterion,optimizer,trainloader,epochs):
    log.info('Training model on {} epochs'.format(epochs))

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            imgs, labels = data

            optimizer.zero_grad()
            
            outputs = network(imgs)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss

            if i % 2000 == 1999:    
                tqdm.write('[{},{}] loss: {}'.format(epoch + 1,i + 1,running_loss / 2000))
                log.info('[{},{}] loss: {}'.format(epoch + 1,i + 1,running_loss / 2000))
                running_loss = 0.0

    log.info('Training finished')
    return network

def test(network,testloader):
    log.info('Testing network performance')
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(tqdm(testloader)):
            images, labels = data 
            outputs = network(images)
            _,predicted = torch.max(outputs.data,1)
            correct += (predicted == labels).sum().item()
    performance = round((correct / (testloader.batch_size*len(testloader)))*100,2)
    tqdm.write('NETWORK PERFORMANCE : {} %'.format(performance))
    log.info('NETWORK PERFORMANCE : {} %'.format(performance))
    return performance