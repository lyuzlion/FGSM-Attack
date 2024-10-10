import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import tqdm
import numpy as np
from models.CNN import *
import argparse
from utils.utils import *


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=75,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='momentum')

    opt = parser.parse_args()
    return opt


def set_loader(opt):
    train_data = torchvision.datasets.MNIST(
        root ='../data/mnist',
        train = True,
        transform = train_transform,
        download = True
    )

    valid_data = torchvision.datasets.MNIST(
        root ='../data/mnist',
        train = False,
        transform = valid_transform,
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size , shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader, valid_loader


def main():
    opt = parse_option()
    
    train_loader, valid_loader = set_loader(opt)
    
    model = CNN().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    criterion = nn.CrossEntropyLoss()
    valid_loss_min = np.Inf

    for epoch in tqdm(range(1, opt.epoch + 1)):
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            output=model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        model.eval()
        valid_loss = 0.0
        correct = 0
        test_num = 0
        for step, (batch_x, batch_y) in enumerate(valid_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            output = model(batch_x)
            loss = criterion(output, batch_y)
            valid_loss += loss.item() * batch_x.size(0)
            y_pred = torch.argmax(output, 1)
            correct += sum(y_pred == batch_y.cuda).item()
            test_num += batch_y.size(0)
        print('now epoch :  ', epoch, '     |   accuracy :   ' , correct / test_num)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), 'checkpoints/cnn_mnist.pt')
            valid_loss_min = valid_loss




if __name__ == '__main__':
    main()

