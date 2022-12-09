from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import Conv2d, Linear, ConvDecoder, DenseDecoder
from torch import Tensor
from torch.autograd import Variable

class ConvNetIDS(nn.Module):
    def __init__(self, width=1, init_type='random', compress_bias = False, vanilla=False,\
                 mode='fan_out', boundary=10, no_shift=False):
        super(ConvNetIDS, self).__init__()

        # TODO: Add weight decoders initialization
        weight_decoders = {}
        bias_decoders = {}
        max_fan = 64*width # Revisar o valor dessa variavel
        var = 24.0/max_fan/((2*boundary+1)**2-1)
        weight_decoders['conv5x5'] = ConvDecoder(25,init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
        weight_decoders['dense'] = DenseDecoder(init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
        groups = ['conv5x5', 'dense']
        for group in groups:
            bias_decoders[group] =  DenseDecoder(init_type, np.sqrt(var), no_shift) if compress_bias and not vanilla \
                                     else nn.Identity()

        self.weight_decoders = weight_decoders
        self.bias_decoders = bias_decoders

        self.feature_extraction_layer = nn.Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same', weight_decoder=weight_decoders['conv5x5'], bias_decoder=bias_decoders['conv5x5']),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same', weight_decoder=weight_decoders['conv5x5'], bias_decoder=bias_decoders['conv5x5']),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.binary_classification_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            Linear(in_features=20416, out_features=64, weight_decoder=weight_decoders['dense'], bias_decoder=bias_decoders['dense'],\
                         compress_bias = compress_bias),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            Linear(in_features=64, out_features=1, weight_decoder=weight_decoders['dense'], bias_decoder=bias_decoders['dense'],\
                         compress_bias = compress_bias)
        )

    def forward(self, x):
        x = x.float()
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer(x)
        x = torch.sigmoid(x)
        return x

    def apply_straight_through(self, use_straight_through=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.use_straight_through = use_straight_through

    def apply_compress_bias(self, compress_bias=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.compress_bias = compress_bias

    def get_weights(self) -> List[Tensor]:
        weights = {}
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                weight = m.weight
                group_name = self.get_group_name(m.weight)
                if group_name == 'dense':
                    weight_reshaped = weight.reshape(weight.size(0)*weight.size(1),1)
                else:
                    weight_reshaped = weight.reshape(weight.size(0)*weight.size(1),weight.size(2)*weight.size(3))
                if group_name in weights:
                    weights[group_name] = torch.cat((weights[group_name],weight_reshaped))
                else:
                    weights[group_name] = weight_reshaped
        return weights

    def get_biases(self) -> List[Tensor]:
        biases = {}
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                bias = m.bias
                group_name = self.get_group_name(m.weight)
                if group_name in biases and bias is not None:
                    biases[group_name] = torch.cat((biases[group_name],bias))
                elif bias is not None:
                    biases[group_name] = bias
        return biases

    def get_group_name(self, param):
        if param.dim()==2:
            return 'dense'
        else:
            h = param.size(2)
            w = param.size(3)
            return f'conv{h}x{w}'
