import torch.nn as nn
import torch
import torch.nn.functional as F
 

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self,x):
        x = self.double_conv(x)
        return x
 
 
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels,out_channels)
        )
 
    def forward(self,x):
        x = self.downsampling(x)
        return x
 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_x = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_x = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        x1 = avg_x + max_x
        out = x * self.sigmoid(x1)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, 1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        out = x * self.sigmoid(x2)
        return out

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_planes, ratio)
        self.SpatialAttention = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.ChannelAttention(x)
        x2 = self.SpatialAttention(x1)
        out = x * self.sigmoid(x2)
        return out

class fgfs1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(fgfs1, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, padding=1, dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x1):
        q1 = self.con1(x1)
        q1 = self.con2(q1)
        q1 = self.con3(q1)
        return q1

class fgfs2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(fgfs2, self).__init__()
        self.con11 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.con22 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.con33 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        r2 = self.con11(x)
        r2 = self.con22(r2)
        r2 = self.con33(r2)
        return r2

class co1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(co1, self).__init__()
        self.co1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.co1(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upsampling(x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
