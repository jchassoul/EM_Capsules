#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matrix Capsules with EM Routing
https://openreview.net/pdf?id=HJWLfGWRb

PyTorch implementation by Huang Zhen @ MultimediaGroup USTC
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from ConvCapsuleLayer import ConvCapsule
from ClassCapsuleLayer import ClassCapsule

NUM_CLASSES = 10

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.Conv1 = ConvCapsule(1,0,32,0,5,2)
        self.PrimaryCaps = ConvCapsule(32,0,32,16,1,1)
        self.ConvCaps1 = ConvCapsule(32,16,32,16,3,2,routing=3)
        self.ConvCaps2 = ConvCapsule(32,16,32,16,3,1,routing=3)
        self.ClassCaps = ClassCapsule(32,16,10,16,routing=3)
        
    def forward(self, x, lamda):
        x = F.relu(self.Conv1(x, lamda))
        x = F.relu(self.PrimaryCaps(x, lamda))
        x = F.sigmoid(self.ConvCaps1(x, lamda))
        x = self.ConvCaps2(x, lamda)
        x = self.ClassCaps(x, lamda)
        return x

def SpreadLoss(self, output, target, m):
    one_shot_target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
    a_t = torch.sum(output * one_shot_target, dim=1)
    loss = torch.sum(max(m - (a_t -output))**2, dim=1) - m**2
    return loss
model = CapsuleNet()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch, m, lamda):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, lamda)
        loss = SpreadLoss(output, target, m)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(m, lamda):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, lamda)
        test_loss += SpreadLoss(output, target, m).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    m = 0.2 + (0.9 - 0.2) * (epoch - 1) / args.epochs
    start_lamda = 0.1
    end_lamda = 0.9
    lamda = start_lamda + (end_lamda - start_lamda) * (epoch - 1) / args.epochs
    train(epoch, m, lamda)
    test(m, lamda)
