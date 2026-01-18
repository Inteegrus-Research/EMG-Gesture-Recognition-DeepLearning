import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class NeuroCNN(nn.Module):
    def __init__(self,n_channels=8,n_classes=5,dropout=0.3):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(n_channels,32,5,padding=2,bias=False),
            nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1,bias=False),
            nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1,bias=False),
            nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(128,n_classes)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.features(x)
        x=self.gap(x).flatten(1)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,dropout=0.2):
        super().__init__()
        self.conv1=nn.Conv1d(in_ch,out_ch,3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm1d(out_ch)
        self.conv2=nn.Conv1d(out_ch,out_ch,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm1d(out_ch)
        self.dropout=nn.Dropout(dropout)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_ch!=out_ch:
            self.shortcut=nn.Sequential(nn.Conv1d(in_ch,out_ch,1,stride=stride,bias=False),
                                        nn.BatchNorm1d(out_ch))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.dropout(out)
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        return F.relu(out)

class NeuroResNet(nn.Module):
    def __init__(self,n_channels=8,n_classes=5,base_filters=32,dropout=0.2):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(n_channels,base_filters,5,padding=2,bias=False),
                                nn.BatchNorm1d(base_filters),nn.ReLU(),nn.MaxPool1d(2))
        self.layer1=ResidualBlock(base_filters,base_filters,1,dropout)
        self.layer2=ResidualBlock(base_filters,base_filters*2,2,dropout)
        self.layer3=ResidualBlock(base_filters*2,base_filters*4,2,dropout)
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(base_filters*4,n_classes)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.stem(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.gap(x).flatten(1)
        return self.fc(x)

class Chomp1d(nn.Module):
    def __init__(self,chomp_size):
        super().__init__()
        self.chomp_size=chomp_size
    def forward(self,x):
        return x[:,:,:-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self,n_inputs,n_outputs,kernel_size,stride,dilation,padding,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(weight_norm(nn.Conv1d(n_inputs,n_outputs,kernel_size,stride=stride,padding=padding,dilation=dilation)),
                               Chomp1d(padding),nn.ReLU(),nn.Dropout(dropout),
                               weight_norm(nn.Conv1d(n_outputs,n_outputs,kernel_size,stride=stride,padding=padding,dilation=dilation)),
                               Chomp1d(padding),nn.ReLU(),nn.Dropout(dropout))
        self.downsample=nn.Conv1d(n_inputs,n_outputs,1) if n_inputs!=n_outputs else None
        self.relu=nn.ReLU()
    def forward(self,x):
        res=x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x)+res)

class NeuroTCN(nn.Module):
    def __init__(self,n_channels=8,n_classes=5,num_channels=[32,32,32,64,64,128,128],kernel_size=3,dropout=0.2):
        super().__init__()
        layers=[]
        for i,out_ch in enumerate(num_channels):
            dilation=2**i
            in_ch=n_channels if i==0 else num_channels[i-1]
            layers.append(TemporalBlock(in_ch,out_ch,kernel_size,1,dilation,(kernel_size-1)*dilation,dropout))
        self.tcn=nn.Sequential(*layers)
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(num_channels[-1],n_classes)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.tcn(x)
        x=self.gap(x).flatten(1)
        return self.fc(x)

