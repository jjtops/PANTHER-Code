import torch
import torch.nn as nn




class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.TConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, "same"),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels*2, 3, 1, "same"),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.TConv(x)

class DoubleConv3Dv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.DConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1,"same"),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, "same"),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.DConv(x)


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv3D(1, 32)
        self.down2 = DoubleConv3D(64, 64)
        self.down3 = DoubleConv3D(128, 128)
        self.down4 = DoubleConv3D(256, 256)
        self.up1 = DoubleConv3Dv2(768, 256)
        self.up2 = DoubleConv3Dv2(384, 128)
        self.up3 = DoubleConv3Dv2(192, 64)


        self.pool = nn.MaxPool3d(2,2)
        self.conv = nn.Conv3d(64, 1,1, 1, "same")

        self.upConv1 =  nn.ConvTranspose3d(512, 512, 2, 2)
        self.upConv2 =  nn.ConvTranspose3d(256, 256, 2,2)
        self.upConv3 =  nn.ConvTranspose3d(128, 128, 2,2)


    def forward(self, x):
        # 1->32, 32->64
        x = self.down1(x)
        copy1 = x
        x = self.pool(x)

       # 64->64, 64->128
        x = self.down2(x)
        copy2 = x
        x = self.pool(x)

        # 128->128, 128->256
        x = self.down3(x)
        copy3 = x
        x = self.pool(x)

        # 256->256, 256->512
        x = self.down4(x)
        x = self.upConv1(x) # 512->512
        x = torch.cat([copy3, x], dim=1) # 256+512
        x = self.up1(x) #768->256, 256->256

        x = self.upConv2(x) # 256->256
        x = torch.cat([copy2, x], dim=1) # 128+256
        x = self.up2(x) # 384->128, 128->128

        x = self.upConv3(x) # 128->128
        x = torch.cat([copy1, x], dim=1) # 64+128
        x = self.up3(x) # 192->64, 64->64

        output = self.conv(x) # 64->1
        return output

# model = UNet3D()
# img = torch.randn(1, 1, 125, 448, 448)
# output = model(img)
# print(output.shape)











