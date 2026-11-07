"""Small U-Net trained from scratch"""
import torch
import torch.nn as nn

# Input: (B, 3, 256, 256)
# Conv2D:
# (B, 32, 256, 256)
# 
# BatchNorm + ReLU:
# (B, 32, 256, 256)
# 
# Conv2D:
# (B, 32, 256, 256)
# 
# BatchNorm + ReLU:
# (B, 32, 256, 256)
# 
# Output: (B, 32, 256, 256)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]     # Initial number of feature channels

        # Encoder (Downsample)
        self.enc1 = DoubleConv(in_channels, c[0])               
        self.enc2 = DoubleConv(c[0], c[1])                      
        self.enc3 = DoubleConv(c[1], c[2])                      
        self.enc4 = DoubleConv(c[2], c[3])                      
        self.bottleneck = DoubleConv(c[3], c[4])                
        self.pool = nn.MaxPool2d(2)                             

        # Decoder (Upsample)                                                    
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)  
        self.dec4 = DoubleConv(c[3] * 2, c[3])                  
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2] * 2, c[2])                  
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[1] * 2, c[1])                  
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[0] * 2, c[0])                  

        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                       # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))                           # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))                           # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))                           # (B, 256, 16, 16)
        b = self.bottleneck(self.pool(e4))                      # (B, 512, 16, 16)
                                                                #         |
                                                                #         v
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))     # (B, 256, 32, 32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))    # (B, 128, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))    # (B, 64, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))    # (B, 32, 256, 256)
        return self.head(d1)   # logits (B, num_classes, H, W)    (B, 4, 256, 256)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)