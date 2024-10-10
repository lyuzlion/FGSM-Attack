import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from models.CNN import *
from utils.FGSM import FGSM_attack
from utils.utils import *
import torch.utils.data as Data
import torchvision
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




def attack(opt, eps):
    test_loader = set_loader(opt)
    model = CNN().cuda()
    model.load_state_dict(torch.load('checkpoints/cnn_mnist.pt', weights_only=False))
    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()

    correct = 0
    adv_samples = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        batch_x.requires_grad = True

        output = model(batch_x)
        init_pred = torch.argmax(output, dim=1)
        
        if not torch.equal(init_pred, batch_y):
            continue

        loss = criterion(output, batch_y)

        model.zero_grad()
        loss.backward()

        adv_sample = FGSM_attack(batch_x, eps)
        output = model(adv_sample)

        final_pred = torch.argmax(output, dim=1)

        if torch.equal(final_pred, batch_y):
            correct += 1
            if (eps == 0) and (len(adv_samples) < 5):
                adv_samples.append((init_pred.item(), final_pred.item(), adv_sample.squeeze().detach().cpu().numpy()))
        else:
            if len(adv_samples) < 5:
                adv_samples.append((init_pred.item(), final_pred.item(), adv_sample.squeeze().detach().cpu().numpy()))
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))

    return final_acc, adv_samples





def main():
    opt = parse_option()
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    examples = []

    for eps in tqdm(epsilons):
        acc, ex = attack(opt, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.6, step=0.1))
    plt.title("FGSM on CNN (MNIST)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    cnt = 0

    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
