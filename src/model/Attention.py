import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
from model import count_parameters

def create_model(params, no_grad=False):
    model = DecisionLevelSingleAttention(128,
                                         params['n_classes'],
                                         params['n_layers'],
                                         128,
                                         params['drop_rate'])
    model = model.cuda()
    if no_grad:
        for param in model.parameters():
            param.detach_().requires_grad_(False)
    return model

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class Attention(nn.Module):
    def __init__(self, n_in, n_out):
        super(Attention, self).__init__()

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = torch.sigmoid(att)

        cla = self.cla(x)
        cla = torch.sigmoid(cla)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)
        
        x = F.hardtanh(x, 0., 1.)
        return x

#https://github.com/pytorch/pytorch/issues/499#issuecomment-503962218
class LocalLinear(nn.Module):
    def __init__(self, in_features, local_features, kernel_size, padding=0, stride=1, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fold_num = (in_features + 2*padding - self.kernel_size)//self.stride + 1
        self.weight = nn.Parameter(torch.randn(fold_num, kernel_size, local_features))
        self.bias = nn.Parameter(torch.randn(fold_num, local_features)) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x, [self.padding]*2, value=0)
        x = x.unfold(-1, size=self.kernel_size, step=self.stride)
        x = torch.matmul(x.unsqueeze(2), self.weight).squeeze(2) + self.bias
        return x
    
class LabelWiseHead(nn.Module):
    def __init__(self, n_labels, hidden):
        super(LabelWiseHead, self).__init__()
        self.n_labels = n_labels
        self.hidden = hidden
        self.weight = nn.Parameter(torch.randn(n_labels, hidden))
        self.bias = nn.Parameter(torch.randn(n_labels))
        
    def forward(self, x):
        out = x.squeeze() * self.weight.view(1, self.n_labels, self.hidden, 1)
        out = out.sum(2) + self.bias.view(1, self.n_labels, 1)
        return out
        
        
class MultiAttention(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super(MultiAttention, self).__init__()
        
        self.classes_num = n_out
        self.hidden = hidden
        self.per_label_hidden = nn.Conv2d(n_in, n_out * hidden, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)
        
        self.att = LabelWiseHead(n_out, hidden)

        self.cla = LabelWiseHead(n_out, hidden)

        self.init_weights()

    def init_weights(self):
        init_layer(self.per_label_hidden)
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        
        embed = F.relu(self.per_label_hidden(x))
        # (samples_num, classes_num*hidden, time_steps, 1)
        N, C, T, _ = embed.shape
        x = embed.view(N, self.classes_num, self.hidden, T, 1)
        # (samples_num, classes_num, hidden, time_steps, 1)

        att = self.att(x)
        att = torch.sigmoid(att)

        cla = self.cla(x)
        cla = torch.sigmoid(cla)

#         att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
#         cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)
        
        x = F.hardtanh(x, 0., 1.)
        return x

class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate, batch_norm=True):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        if batch_norm:
            self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
            in_channels = freq_bins if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            if batch_norm:
                self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        if batch_norm:
            self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        for bn in self.batchnorm:
            init_bn(bn)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        out = self.batchnorm[0](x)
        residual = x
        all_outs = [out]

        for i in range(len(self.conv1x1)):
            out = F.dropout(F.relu(self.batchnorm[i+1](self.conv1x1[i](out))),
                            p=drop_rate,
                            training=self.training)
            all_outs.append(out)
        out = out + residual
        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return out

        else:
            return all_outs

class DecoderLayers(nn.Module):

    def __init__(self, z_dim, emb_layers, hidden_units, drop_rate):
        super(DecoderLayers, self).__init__()

        self.z_dim = z_dim
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        # self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
            in_channels = z_dim if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            # self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        # self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        # for bn in self.batchnorm:
        #     init_bn(bn)

    def forward(self, input):
        drop_rate = self.drop_rate
        # out = self.batchnorm[0](input)
        out = input
        for i in range(len(self.conv1x1)):
            # out = F.dropout(F.relu(self.batchnorm[i+1](self.conv1x1[i](out))),
            #                 p=drop_rate,
            #                 training=self.training)
            out = F.dropout(F.relu(self.conv1x1[i](out)),
                            p=drop_rate,
                            training=self.training)

        return out

class DecisionLevelSingleAttention(nn.Module):
    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate, batch_norm=True):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(
            freq_bins=freq_bins,
            emb_layers=emb_layers,
            hidden_units=hidden_units,
            drop_rate=drop_rate,
            batch_norm=batch_norm)

        self.attention = Attention(
            n_in=hidden_units,
            n_out=classes_num)
            
        self.param_count = count_parameters(self)
        print(self.param_count)

    def init_weights(self):
        pass

    def forward(self, X):
        """X: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(X)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output
