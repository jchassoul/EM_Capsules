#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Capsules with EM Routing
https://openreview.net/pdf?id=HJWLfGWRb

PyTorch implementation by Huang Zhen @ MultimediaGroup USTC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

from torch.nn import init
import pdb

COORDINATE_SCALE = 10.

class ClassCapsule(nn.Module):
    def __init__(self, 
                 in_channel, in_dim,
                 classes, out_dim, routing):
        super(ClassCapsule, self).__init__()
        
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classes = classes
        self.routing = routing
        
        self.beta_v = Variable(torch.randn(self.classes)).cuda()
        self.beta_a = Variable(torch.randn(self.classes)).cuda()
        self.capsules = nn.Conv2d(in_channels=in_channel * in_dim,
                                  out_channels=in_channel * out_dim * classes,
                                  kernel_size=1,
                                  stride=1,
                                  groups=in_channel)
        
    def EM_routing(self, votes, activations):
        # routing coefficient
        R = (1. / self.classes) * Variable(torch.ones(self.batches, self.in_channel, self.classes, self.h, self.w), requires_grad=False).cuda()
        # activations.size = [b, in_channel, h, w]
        activations = activations.squeeze(dim=2)
        votes_reshape = votes.view(self.batches, self.in_channel, self.classes, self.out_dim, self.h, self.w)
        for _ in range(self.routing):
            # r_hat.size = [b, in_c, classes, h, w]
            r_hat = R * activations[:,:,None,:,:]
            # sum_r_hat.size = [b, classes]
            sum_r_hat = r_hat.sum(4).sum(3).sum(1)
            # votes_reshape.size = [b, in_channel, classes, out_dim, h, w]
            # u_h.size = [b, classes, out_dim]
            u_h = torch.sum(r_hat[:,:,:,None,:,:] * votes_reshape, dim=5).sum(4).sum(1) / sum_r_hat[:,:,None]
            sigma_h_square = torch.sum(r_hat[:,:,:,None,:,:]*(votes_reshape - u_h[:,None,:,:,None,None])**2, dim=5).sum(4).sum(1) / sum_r_hat[:,:,None]
            # cost_h.size = [b, classes, out_dim]
            cost_h = (self.beta_v[None,:,None] + torch.log(sigma_h_square)) * sum_r_hat[:,:,None]
            # a_hat.size = [b,classes]
            a_hat = torch.sigmoid(self.lamda * (self.beta_a[None,:] - torch.sum(cost_h, dim=2)))
            
            sigma_product = Variable(torch.ones(self.batches, self.classes), requires_grad=False).cuda()
            for dm in range(self.out_dim):
                sigma_product = 2 * 3.1416 * sigma_product * sigma_h_square[:,:,dm]
            # p_c.size = [b, in_channel, classes, h, w]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:,None,:,:,None,None])**2 / (2 * sigma_h_square[:,None,:,:,None,None]), dim=3)) / torch.sqrt(sigma_product[:,None,:,None,None])
            R = a_hat[:,None,:,None,None] * p_c / torch.sum(a_hat[:,None,:,None,None] * p_c, dim=2, keepdim=True)
        return a_hat, u_h

    def CoordinateAddition(self, vector):
        output = Variable(torch.zeros(vector.size())).cuda()
        coordinate_x = Variable(torch.FloatTensor(torch.arange(0, self.h))/COORDINATE_SCALE, requires_grad=False).cuda()
        coordinate_y = Variable(torch.FloatTensor(torch.arange(0, self.w))/COORDINATE_SCALE, requires_grad=False).cuda()
        output[:,:,0,:,:] = vector[:,:,0,:,:] + coordinate_x[None,None,:,None]
        output[:,:,1,:,:] = vector[:,:,1,:,:] + coordinate_y[None,None,None,:]
        if output.size(2) >2:
            output[:,:,2:,:,:] = vector[:,:,2:,:,:]
        return output
    
    def forward(self, x, lamda=0):
        self.lamda = lamda
        size = x.size()
        self.batches = size[0]
        self.h = size[2]
        self.w = size[3]
        x_reshape = x.view(size[0], self.in_channel, 1+self.in_dim, size[2], size[3])
        activations = x_reshape[:,:,0,:,:]
        vector = x_reshape[:,:,1:,:,:]
        vec = self.CoordinateAddition(vector)
        
        
#        for i in range(self.h):
#            for j in range(self.w):
#                vector[:,:,0,i,j] += i/10.
#                vector[:,:,1,i,j] += j/10.
        # vector.size = [b,in_channel*in_dim, h,w]
        vec = vec.view(size[0], -1, size[2], size[3]) 
        # votes.size = [b, in_channel*out_dum*classes, h,w]
        votes = self.capsules(vec)
        # output_a.size = [b, classes]
        # output_v.size = [b, classes, out_dim]
        output_a, output_v = self.EM_routing(votes, activations)
        return output_a

def main():
    torch.cuda.manual_seed(1)
    layer = ClassCapsule(2, 2, 2, 2, 3)
    layer.cuda()
    x = Variable(torch.rand(3,6,5,5))
    lamda=Variable(torch.rand(1))
    y = layer(x.cuda(), lamda.cuda())
    print(y.size())
    
if __name__ == '__main__':
    main()
