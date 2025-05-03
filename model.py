import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import torch

class SRnet_dtcwt(nn.Module):
    def __init__(self, biort='near_sym_b', qshift='qshift_b', device='cuda'):
        super(SRnet_dtcwt, self).__init__()
        # 定义小波变换与小波逆变换类
        self.wt_level2=DTCWTForward(J=2, biort=biort, qshift=qshift, include_scale=[True, False])
        self.iwt=DTCWTInverse(biort=biort, qshift=qshift)

        self.device = device

        self.conv28_64 = nn.Sequential(
            nn.Conv2d(28, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv64_16 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, bias=True),
        )
        self.ca64 = ChannelAttention(in_channels=64, reduction=4)
        self.RRDB = RRDB(nf=64, gc=64)

        # 在初始化中添加
        self.dense_block = DenseBlock(in_channels=64, growth_rate=16, num_layers=4)
        self.transition = TransitionLayer(64 + 16*4, 64)  # 从dense_block输出转换回64通道

    def forward(self, LR):
        # LR.shape = (batch_size, 1, height, width)
        batch_size, _, height, width = LR.shape

        # LR: (batch_size, 1, height, width) -> (batch_size, 1, height*2, width*2)
        SR_bicubic = nn.functional.interpolate(LR/255.0, scale_factor=2, mode='bicubic', align_corners=False)

        # wavelet transform
        with torch.amp.autocast(self.device,enabled=False):
            low, high=self.wt_level2(SR_bicubic)

        # low1.shape = (batch_size, 1, height*2, width*2)
        # high1.shape = (batch_size, 1, 6, height, width, 2)
        # high2.shape = (batch_size, 1, 6, height/2, width/2, 2)
        low1, high1, high2 = low[0], high[0], high[1]

        # low1: (batch_size, 1, height*2, width*2) -> (batch_size, 4, height, width)
        low1 = nn.PixelUnshuffle(downscale_factor=2)(low1)

        # high1: (batch_size, 1, 6, height, width, 2) -> (batch_size, 12, height, width)
        high1 = high1.permute(0,3,4,1,2,5).reshape(batch_size, height, width, 12).permute(0,3,1,2)

        # high2: (batch_size, 1, 6, height/2, width/2, 2) -> (batch_size, 12, height/2, width/2) -> (batch_size, 12, height, width)
        high2 = high2.permute(0,3,4,1,2,5).reshape(batch_size, height//2, width//2, 12).permute(0,3,1,2)
        high2 = nn.functional.interpolate(high2, scale_factor=2, mode='bicubic', align_corners=False)
        
        out = torch.cat((low1, high1, high2), dim=1)
        early_features = self.conv28_64(out)

        # RRDB
        out = self.ca64(early_features)
        out = self.RRDB(out)
        out = self.RRDB(out)

        # 密集特征提取
        out = self.ca64(out)
        dense_features = self.dense_block(out)
        out = self.transition(dense_features) + early_features

        # 还原通道
        out = self.conv64_16(out)

        pred_low, pred_high = out[:, 0:4, :, :], out[:, 4:16, :, :]
        
        # pred_low: (batch_size, 4, height, width) -> (batch_size, 1, height*2, width*2)
        pred_low = nn.PixelShuffle(upscale_factor=2)(pred_low)

        # pred_high: (batch_size, 12, height, width) -> (batch_size, 1, 6, height, width, 2)
        pred_high = pred_high.permute(0, 2, 3, 1).reshape(batch_size, height, width, 6, 2).permute(0, 3, 1, 2, 4).unsqueeze(1)
        
        # iwt
        with torch.amp.autocast(self.device, enabled=False):
            SR = self.iwt((pred_low, [pred_high]))
        
        SR += SR_bicubic
        return SR*255.0

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        # gc: 每个卷积层产生的新特征通道数
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # 残差缩放

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # 残差in残差结构
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
        
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
    def forward(self, x):
        return self.act(self.conv(x))