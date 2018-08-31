from typing import List, Any
import torch
import torchvision
import torch.backends.cudnn as cudnn
import os
import sys
import argparse
import numpy as np
from res164 import *
from wide_resnet import *
use_cuda = torch.cuda.is_available()
batch_size = 8
criterion = torch.nn.CrossEntropyLoss()
mean_cifar100 = (0.50707516, 0.48654887, 0.44091784)
std_cifar100 = (0.26733429, 0.25643846, 0.27615047)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean_cifar100, std_cifar100)]
)
print("Data Preparation......")
test_data = torchvision.datasets.CIFAR100(
    root='/data/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=1)


print("Resuming from checkPoint....")
checkPoint1 = torch.load('../checkpoint/densenet.t7')
net1 = checkPoint1['net']
acc1 = torch.load('../checkpoint/densenet_acc.t7')['acc']
checkPoint2 = torch.load('../checkpoint/resnet.t7')
net2 = checkPoint2['model']
acc2 = torch.load('../checkpoint/resnet_acc.t7')['acc']
checkPoint3 = torch.load('../checkpoint/wideresnet.t7')
net3 = checkPoint3['net']
acc3 = torch.load('../checkpoint/wideresnet_acc.t7')['acc']


def output(net):
    net.eval()
    outputs = net(inputs)
    return outputs


correct = 0
total = 0
for inputs, targets in test_loader:
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = torch.autograd.Variable(
        inputs), torch.autograd.Variable(targets)
    output4 = []
    output1 = output(net1)
    output2 = output(net2)
    output3 = output(net3)
    for i in range(batch_size):
        _, a = torch.max(output1[i], 0)
        _, b = torch.max(output2[i], 0)
        _, c = torch.max(output3[i], 0)
        a, b, c = a.item(), b.item(), c.item()
        d = acc1[a]
        e = acc2[b]
        f = acc3[c]
        maxi = max(d, e, f)
        # if a==b:
        #	output4.append(a)
        # else:
        #	output4.append(c)
        if d == maxi:
            output4.append(a)
        elif e == maxi:
            output4.append(b)
        else:
            output4.append(c)
    print(output4, targets.data)
    output4 = np.array(output4)
    output4 = torch.from_numpy(output4)
    output4 = output4.cuda()
    output4 = torch.autograd.Variable(output4)
    total += targets.size(0)
    correct += output4.eq(targets.data).cpu().sum()
    acc = 100. * correct / total
    print("\n|acc:%.2f%%" % acc)
    print(total, correct)
print("\n|acc:%.4f%%" % acc)
