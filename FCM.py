from MFE_parts import *
from UNet_FGFS_CBAM_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.cbam = CBAM(1024)

        self.up1 = Up(1024, 512)
        self.fgfs1 = fgfs1(512, 256)
        self.fgfs11 = fgfs2(512, 256)
        self.sigmoid1 = nn.Sigmoid()
        self.co1 = co1(1024, 512)
        self.conv1 = DoubleConv(1536, 512)

        self.up2 = Up(512, 256)
        self.fgfs2 = fgfs1(256, 128)
        self.fgfs22 = fgfs2(256, 128)
        self.sigmoid2 = nn.Sigmoid()
        self.co2 = co1(512, 256)
        self.conv2 = DoubleConv(768, 256)

        self.up3 = Up(256, 128)
        self.fgfs3 = fgfs1(128, 64)
        self.fgfs33 = fgfs2(128, 64)
        self.sigmoid3 = nn.Sigmoid()
        self.co3 = co1(256, 128)
        self.conv3 = DoubleConv(384, 128)

        self.up4 = Up(128, 64)
        self.fgfs4 = fgfs1(64, 32)
        self.fgfs44 = fgfs2(64, 32)
        self.sigmoid4 = nn.Sigmoid()
        self.co4 = co1(128, 64)
        self.conv4 = DoubleConv(192, 64)

        self.d1 = D(192, 192)
        self.d2 = D(576, 576)

        self.u1 = U(576, 384)
        self.one_Conv1 = oneConv(384, 192)
        self.u2 = U(192, 128)
        self.one_Conv2 = oneConv(128, 64)

        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        c1 = self.cbam(x5)
        x6 = self.up1(c1)
        x7 = self.fgfs1(x6)
        x8 = self.fgfs11(x4)
        x9 = self.sigmoid1(x8)
        x = x7 * x9
        x = x + x7
        x = torch.cat([x, x8], dim=1)
        x = self.co1(x)
        x = torch.cat([x6, x, x4], dim=1)
        x = self.conv1(x)

        x13 = self.up2(x)
        x10 = self.fgfs2(x13)
        x11 = self.fgfs22(x3)
        x12 = self.sigmoid2(x11)
        x = x10 * x12
        x = x + x10
        x = torch.cat([x, x11], dim=1)
        x = self.co2(x)
        x = torch.cat([x13, x, x3], dim=1)
        x = self.conv2(x)

        x14 = self.up3(x)
        x15 = self.fgfs3(x14)
        x16 = self.fgfs33(x2)
        x17 = self.sigmoid3(x16)
        x = x15 * x17
        x = x + x15
        x = torch.cat([x, x16], dim=1)
        x = self.co3(x)
        x = torch.cat([x14, x, x2], dim=1)
        x = self.conv3(x)

        x18 = self.up4(x)
        x19 = self.fgfs4(x18)
        x20 = self.fgfs44(x1)
        x21 = self.sigmoid4(x20)
        x = x19 * x21
        x = x + x19
        x = torch.cat([x, x20], dim=1)
        x = self.co4(x)
        x = torch.cat([x18, x, x1], dim=1)
        x = self.conv4(x)

        d1 = self.d1(x)
        d2 = self.d2(d1)
        u1 = self.u1(d2, d1)
        u1 = self.one_Conv1(u1)
        u1 = self.u2(u1, x)
        u1 = self.one_Conv2(u1)
        out = self.out_conv(u1)
        return out

if __name__ == '__main__':
    input = torch.randn(8, 1, 224, 224)
    net = UNet(in_channels=1, num_classes=1)
    output = net(input)
    print(net)
