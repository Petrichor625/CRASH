import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channel * 2, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat((avg_y, max_y), dim=1) 
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=32, w=9):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.channel_attention = ChannelAttention(dim*2) 
        self.conv1x1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1)


    def forward(self, x, spatial_size=None):
        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.conv1x1(x) 
        x = x.view(x.size(0), x.size(2), x.size(1))
        B, N, C = x.shape
        a = 32
        b = 16
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        x_real = x.real.permute(0, 3, 1, 2)
        x_imag = x.imag.permute(0, 3, 1, 2)
        x_combined = torch.cat((x_real, x_imag), dim=1)

        x_attended = self.channel_attention(x_combined)

        x_real, x_imag = torch.chunk(x_attended, 2, dim=1)
        x = torch.complex(x_real, x_imag).permute(0, 2, 3, 1)

        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        x = x.mean(dim=2, keepdim=True)
        return x



if __name__ == '__main__':
    block = SpectralGatingNetwork(3).cuda()
    input = torch.rand(10, 512, 1).cuda()
    output = block(input) 
    print(output.shape)
