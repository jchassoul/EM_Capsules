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

class ConvCapsuleLayer(nn.Module):
    def __init__(self, 
                 in_channel, in_dim,
                 out_channel, out_dim, 
                 kernel_size, stride,
                 routing, lamda):
        super(ConvCapsuleLayer, self).__init__()
        
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.out_dim = out_dim
        self.routing = routing
        self.kernel_size = kernel_size
        self.stride = stride
        self.lamda = lamda
        
        if self.routing:
            self.routing_capsule = nn.Conv2d(in_channels=
                                             kernel_size * kernel_size *
                                             in_dim * in_channel,
                                             out_channels=
                                             kernel_size * kernel_size *
                                             out_dim * in_channel * out_channel,
                                             kernel_size=1,
                                             stride=1,
                                             groups=
                                             kernel_size * kernel_size * in_channel)
            
        else:
            self.no_routing_capsule = nn.Conv2d(in_channel * (in_dim + 1),
                                                out_channel * (out_dim + 1),
                                                kernel_size=kernel_size,
                                                stride=stride)
    
    def squash(self, tensor):
        # no sure about this operation, may cause error
        size = tensor.size()
        if (len(tensor.size()) < 5):
            # [batch, channel, h, w] --> [batch, cap_channel, cap_dim, h, w]
            tensor = torch.stack(tensor.split(self.out_dim, dim=1), dim = 1)
        squared_norm = (tensor ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        outputs = scale * tensor / torch.sqrt(squared_norm)
        return outputs.view(size)
    
    def down_h(self, h):
        return range(h*self.stride, h*self.stride+self.kernel_size)
    
    def down_w(self, w):
        return range(w*self.stride, w*self.stride+self.kernel_size)
            
    def EM_routing(self, votes, activations):
        beta_v = Variable(torch.randn(self.out_channel, self.out_h, self.out_w)).cuda()
        beta_a = Variable(torch.randn(self.out_channel, self.out_h, self.out_w)).cuda()
        
        R = (1. / self.out_channel) * Variable(torch.ones(self.batches, self.in_channel, self.kernel_size, self.kernel_size, 
                                                          self.out_channel, self.out_h, self.out_w)).cuda()
        votes_reshape = votes.view(self.batches, self.in_channel, self.kernel_size, self.kernel_size, 
                                   self.out_channel, self.out_dim, self.out_h, self.out_w)
        activations = activations.squeeze(dim=2)
        
        a_reshape = [activations[:, :, :, self.down_w(w)][:,:,self.down_h(h),:] for h in range(self.out_h) for w in range(self.out_w)]
        a_stack = torch.stack(a_reshape, dim=4).view(self.batches, self.in_channel, self.kernel_size, self.kernel_size, self.out_h, self.out_w)
        for _ in range(self.routing):
            # M-STEP
            # r_hat.size = [b, in_c, k, k, out_c, out_h, out_w]
            r_hat = R * a_stack[:,:,:,:,None,:,:]
            # sum_r_hat.size = [b, out_c, out_h, out_w]
            sum_r_hat = r_hat.sum(3).sum(2).sum(1)
            # u_h.size = [b, out_c, out_d, out_h, out_w]
            u_h = torch.sum(r_hat[:,:,:,:,:,None,:,:] * votes_reshape, dim=3).sum(2).sum(1) / sum_r_hat[:,:,None,:,:]
            # sigma_h_square.size = [b, out_c, out_d, out_h, out_w]
            sigma_h_square = torch.sum(r_hat[:,:,:,:,:,None,:,:] * (votes_reshape - u_h[:,None,None,None,:,:,:,:]) ** 2, dim=3).sum(2).sum(1) / sum_r_hat[:,:,None,:,:]
            # cost_h.size = [b, out_c, out_d, out_h, out_w]
            cost_h = (beta_v[None,:,None,:,:] + torch.log(torch.sqrt(sigma_h_square))) * sum_r_hat[:, :, None, :, :]
            # a_hat.size = [b, out_c, out_h, out_w]
            a_hat = torch.sigmoid(self.lamda * (beta_a[None,:,:,:] - cost_h.sum(2)))
            
            # E-STEP
            # sigma_product.size = [b, out_c, out_h, out_w]
            sigma_product = Variable(torch.ones(self.batches, self.out_channel, self.out_h, self.out_w)).cuda()
            for dm in range(self.out_dim):
                sigma_product = sigma_product * 2 * 3.1416 *sigma_h_square[:,:,dm,:,:]
            # p_c.size = [b, in_c, k, k, out_c, out_h, out_w]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:,None,None,None,:,:,:,:]) ** 2 / (2 * sigma_h_square[:,None,None,None,:,:,:,:]), dim=5) / torch.sqrt(sigma_product[:,None,None,None,:,:,:]))
            # R,size = [b,in_c, k, k, out_c, out_h, out_w]
            R = a_hat[:,None,None,None,:,:,:] * p_c / torch.sum(a_hat[:,None,None,None,:,:,:] * p_c, dim=6, keepdim=True).sum(dim=5, keepdim=True).sum(dim=4, keepdim=True)
        return a_hat, u_h
               
    def forward(self, x):
        if self.routing:
            size = x.size()
            self.batches = size[0]
            out_h = int((size[2] - self.kernel_size) / self.stride) + 1
            out_w = int((size[3] - self.kernel_size) / self.stride) + 1
            self.out_h = out_h
            self.out_w = out_w
            x_reshape = x.view(size[0], self.in_channel, 1+self.in_dim, size[2], size[3])
            activations = x_reshape[:,:,0,:,:]
            vector = x_reshape[:,:,1:,:,:].contiguous().view(size[0], -1, size[2], size[3])
            # sampling
            # z[batch, k*k*vhannel, out_h, out_w]            
            maps = []
            for k_h in range(self.kernel_size):
                for k_w in range(self.kernel_size):
                    onemap = [vector[:, :, k_h+i, k_w+j] for i in range(0, out_h*self.stride, self.stride) for j in range(0, out_w*self.stride, self.stride)]
                    onemap = torch.stack(onemap, dim=2)
                    onemap = onemap.view(size[0], onemap.size(1), out_h, out_w)
                    maps.append(onemap)
            # maps channel is kernal_size**2 * in_channel * in_dim
            map_ = torch.cat(maps, dim=1)
            
            # votes.size: (out_h * out_w) * k * k * in_channel * out_channel( * D)
            votes = self.routing_capsule(map_)
            # output_a.size = [b, out_c, out_h, out_w]
            # output_v.size = [b, out_c, out_d, out_h, out_w]
            output_a, output_v = self.EM_routing(votes, activations)
            outputs = torch.cat([output_a[:,:,None,:,:], self.squash(output_v)], dim=2)
            return outputs.view(self.batches, self.out_channel * (self.out_dim + 1), self.out_h, self.out_w)
        else:
            # outputs [batch, channel, out_h, out_w]
            outputs = self.no_routing_capsule(x)
            return self.squash(outputs)

def main():
    torch.cuda.manual_seed(1)
    layer = ConvCapsuleLayer(2, 2, 2, 2, 3, 2, routing=3, lamda=Variable(torch.rand(1)).cuda())
    layer.cuda()
    x = Variable(torch.rand(2,6,5,5))
    y = layer(x.cuda())
    
if __name__ == '__main__':
    main()
