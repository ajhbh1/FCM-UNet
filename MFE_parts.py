import torch.nn as nn
import torch
import torch.nn.functional as F


class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(oneConv, self).__init__()
        self.one_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.one_Conv(x)
        return x

class twoConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(twoConv, self).__init__()
        self.two_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.two_conv(x)
        return x

class D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D, self).__init__()
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.max7 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        self.conv1 = oneConv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.max3(x)
        x2 = self.max5(x)
        x3 = self.max7(x)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x4)
        return x

class U(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U, self).__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 3, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x




