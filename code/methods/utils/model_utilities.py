
# To restore to what was there originally just copy paste this file's content from github, j a few new functions introduced
import numpy as np
import torch
import torch.nn as nn


def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()
        
    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)
        
    def forward(self, x):
        x = self.double_conv(x)

        return x

class FD(nn.Module):
    def __init__(self, in_channels,factor, 
                kernel_size=(1,7), stride=(1,4), padding=(0,2),
                dilation=1, bias=False):
        super().__init__()
        self.fd=nn.Sequential()
        self.factor=factor
        self.in_channels=in_channels
        self.out_channels=in_channels*2
        for i in range(0,factor):
            self.fd.append(nn.Conv2d(in_channels=self.in_channels, 
                    out_channels=self.out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias))
            self.fd.append(nn.BatchNorm2d(self.out_channels))
            self.fd.append(nn.ReLU(inplace=True))
            self.in_channels=self.out_channels
            self.out_channels=self.in_channels*2
        self.init_weights()
        
    def init_weights(self):
        for layer in self.fd:
            init_layer(layer)
        
    def forward(self, x):
        x = self.fd(x)

        return x
    
class FU(nn.Module):
    def __init__(self, in_channels,factor, 
                kernel_size=(1,1), stride=(1,1), padding=(0,0),
                dilation=1, bias=False):
        super().__init__()
        self.fu=nn.Sequential()
        self.factor=factor
        
        self.out_channels=in_channels//(2*factor)
        
        self.fu.append(nn.Conv2d(in_channels=in_channels, 
                    out_channels=self.out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias))
        self.fu.append(nn.Upsample(scale_factor=(1,4**factor)))

        self.init_weights()
        
    def init_weights(self):
        for layer in self.fu:
            init_layer(layer)
        
    def forward(self, x):
        x = self.fu(x)

        return x

class TFCMConvBlock(nn.Module):
    def __init__(self, in_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                 dilation=1,bias=False):
        super().__init__()
        self.tfcmconv=nn.Sequential()
        self.tfcmconv.append(nn.Conv2d(in_channels=in_channels, 
                    out_channels=in_channels,
                    kernel_size=(1,1), stride=stride,
                    padding=(0,0), dilation=1, bias=bias))
        self.tfcmconv.append(nn.Conv2d(in_channels=in_channels, 
                    out_channels=in_channels,
                    kernel_size=(1,1), stride=stride,
                    padding=(0,0), dilation=1, bias=bias))
        self.tfcmconv.append(nn.Conv2d(in_channels=in_channels, 
                    out_channels=in_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding="same", dilation=(dilation,1), bias=bias))
    

        self.init_weights()
        
    def init_weights(self):
        for layer in self.tfcmconv:
            init_layer(layer)
        
    def forward(self, x):
        
        x = self.tfcmconv(x)
        return x

class TFCM(nn.Module):
    def __init__(self, in_channels,m=6):
        super().__init__()
        self.tfcm=nn.Sequential()
        for i in range(0,m):
            self.tfcm.append(TFCMConvBlock(in_channels=in_channels,dilation=2**i))
        
    def forward(self, x):
        x = self.tfcm(x)

        return x

class MFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FD12=FD(in_channels=in_channels,factor=1)
        self.FD13=FD(in_channels=in_channels,factor=2)
        self.FD23=FD(in_channels=2*in_channels,factor=1)
        self.FU21=FU(in_channels=2*in_channels,factor=1)
        self.FU31=FU(in_channels=4*in_channels,factor=2)
        self.FU32=FU(in_channels=4*in_channels,factor=1)
        self.TFCM1_1=TFCM(in_channels=in_channels)
        self.TFCM1_2=TFCM(in_channels=in_channels)
        self.TFCM1_3=TFCM(in_channels=in_channels)
        self.TFCM2_1=TFCM(in_channels=in_channels*2)
        self.TFCM2_2=TFCM(in_channels=in_channels*2)
        self.TFCM2_3=TFCM(in_channels=in_channels*2)
        self.TFCM3_1=TFCM(in_channels=in_channels*4)
        self.TFCM3_2=TFCM(in_channels=in_channels*4)

        
    def forward(self, x):
        
        x1stage1=self.TFCM1_1(x)
        x2stage1=self.TFCM2_1(self.FD12(x))
        
        x1stage1sum=x1stage1+self.FU21(x2stage1)
        x2stage1sum=self.FD12(x1stage1)+x2stage1
        x1stage2=self.TFCM1_2(x1stage1sum)
        x2stage2=self.TFCM2_2(x2stage1sum)
        x3stage2=self.TFCM3_1(self.FD23(x2stage1sum))
        x1stage2sum=x1stage2+ self.FU21(x2stage2)+self.FU31(x3stage2)
        x2stage2sum=x2stage2+self.FD12(x1stage2)+self.FU32(x3stage2)
        x3stage2sum=x3stage2+self.FD13(x1stage2)+self.FD23(x2stage2) 
        x1stage3=self.TFCM1_3(x1stage2sum)
        x2stage3=self.TFCM2_3(x2stage2sum)
        x3stage3=self.TFCM3_2(x3stage2sum)
        xfinal=x1stage3+self.FU21(x2stage3)+self.FU31(x3stage3)
        return xfinal
       
class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)

