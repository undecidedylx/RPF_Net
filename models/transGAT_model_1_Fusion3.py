import torch
import torch.nn as nn
 
 
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=1):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,bias=False)
                              # padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x).unsqueeze(3))
        avg_out = self.mlp(self.avg_pool(x).unsqueeze(3))
        channel_out = self.sigmoid(max_out + avg_out).squeeze(3)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1).unsqueeze(3))).squeeze(3)
        x = spatial_out * x
        return x
 
x = torch.randn(1,1000,1024)
net = CBAMLayer(1000)
y = net.forward(x)
print(y.shape)