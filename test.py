import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from models.CNN import *
from utils.utils import *
import argparse
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')

    opt = parser.parse_args()
    return opt


def set_loader(opt):
    test_data = torchvision.datasets.MNIST(
        transform=test_transform,
        root='../data/mnist',
        train=False,
    )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=True)
    return test_loader

def main():
    opt = parse_option()
    test_loader = set_loader(opt)
    model = CNN().cuda()
    model.load_state_dict(torch.load('checkpoints/cnn_mnist.pt', weights_only=False))

    model.eval()
    correct = 0
    test_num = 0
    for step, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
        output = model(batch_x.cuda())
        y_pred = torch.argmax(output, 1)
        correct += sum(y_pred == batch_y.cuda()).item()
        test_num += batch_y.size(0)
    print('Accuracy :   ' , correct / test_num)


if __name__ == '__main__':
    main()