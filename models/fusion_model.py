import torch
import torch.nn as nn
import torch.nn.parallel
from flgc import Flgc2d,Flgc2d_share
from torch.autograd import Variable
from SCNet import *
from Sptransformer import *
from DWT_IDWT_layer import *
from pytorch_wavelets import DWTForward, DWTInverse,DTCWTForward, DTCWTInverse
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU()
                                     #Flgc2d(1, 32, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(32),
                                     #nn.ReLU()
                                     )
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU()
                                     #Flgc2d(32, 64, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(64),
                                     #nn.ReLU()
                                     )
        self.conv1_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU()
                                     #Flgc2d(64, 128, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(128),
                                     #nn.ReLU()
                                     )
        self.conv2_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU()
                                     #Flgc2d(64, 128, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(128),
                                     #nn.ReLU()
                                     )
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU()
                                     #Flgc2d(64, 128, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(128),
                                     #nn.ReLU()
                                     )
        self.conv2_3 = nn.Conv2d(32, 1, 3, 1, 0)
        #self.layer2 = nn.Linear(128*64*64, 64*64)
        #self.layer3 = nn.Sequential(nn.Linear(512, 1))

    def forward(self, infrared, visible):
        input = torch.cat((infrared, visible), dim=1)
        conv1_1 = self.conv1_1(self.padding1(input))
        conv1_2 = self.conv1_2(self.padding1(conv1_1))
        conv1_2 = torch.cat((conv1_2, conv1_1), dim=1)
        conv1_3 = self.conv1_3(self.padding1(conv1_2))
        conv1_3 = torch.cat((conv1_2, conv1_3), dim=1)
        conv2_1 = self.conv2_1(self.padding1(conv1_3))
        conv2_2 = self.conv2_2(self.padding1(conv2_1))
        conv2_3 = self.conv2_3(self.padding1(conv2_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv2_3
class Fusion_fft_Net(nn.Module):
    def __init__(self):
        super(Fusion_fft_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     #Flgc2d(1, 32, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(32),
                                     #nn.ReLU()
                                     )
        self.conv1_2 = nn.Sequential(FFTConv2d(32, 64, 3, 0, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     #Flgc2d(32, 64, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(64),
                                     #nn.ReLU()
                                     )
        self.conv1_3 = nn.Sequential(FFTConv2d(64, 128, 3, 0, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     #Flgc2d(64, 128, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(128),
                                     #nn.ReLU()
                                     )
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     #Flgc2d(1, 32, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(32),
                                     #nn.ReLU()
                                     )
        self.conv2_2 = nn.Sequential(FFTConv2d(32, 64, 3, 0, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     #Flgc2d(32, 64, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(64),
                                     #nn.ReLU()
                                     )
        self.conv2_3 = nn.Sequential(FFTConv2d(64, 128, 3, 0, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     #Flgc2d(64, 128, 1, 1, 0, 1, groups=8, bias=False),
                                     #nn.BatchNorm2d(128),
                                     #nn.ReLU()
                                     )
        self.conv3_1 = nn.Sequential(FFTConv2d(256, 128, 3, 0, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU()
                                     #Flgc2d(256,128,1,1,0,bias=False, groups=1)
                                     )
        self.conv3_2 = nn.Sequential(FFTConv2d(128, 64, 3, 0, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU()
                                     #Flgc2d(128,64,1,1,0,bias=False, groups=1)
                                     )
        self.conv3_3 = nn.Conv2d(64, 1, 3, 1, 0)
                                     #nn.BatchNorm2d(1),
                                     #nn.ReLU(),
                                     #Flgc2d(64,1,1,1,0,bias=False, groups=1)
        #self.conv3_4 = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv1_2 = self.conv1_2(self.padding1(conv1_1))
        conv1_3 = self.conv1_3(self.padding1(conv1_2))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv2_2 = self.conv2_2(self.padding1(conv2_1))
        conv2_3 = self.conv2_3(self.padding1(conv2_2))
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        conv3_1 = self.conv3_1(self.padding1(concate))
        conv3_2 = self.conv3_2(self.padding1(conv3_1))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_shareNet(nn.Module):
    def __init__(self):
        super(Fusion_shareNet, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))

        self.conv_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        self.conv_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))

        self.conv_share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.conv_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.conv_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))

        self.conv_share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))

        self.conv3_1 = nn.Conv2d(512, 256, 3,1,0)
        self.conv3_2 = nn.Conv2d(256, 128, 3,1,0)
        self.conv3_3 = nn.Conv2d(128, 64, 3, 1,0)
        self.conv3_4 = nn.Conv2d(64, 1, 3, 1,0)
    def forward(self, infrared, visible):
        conv1_1 = nn.functional.conv2d(self.padding1(infrared), self.conv_ir_0, bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_1 = nn.functional.conv2d(self.padding1(visible), self.conv_vis_0, bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_1 = nn.functional.relu(conv1_1)
        conv2_1 = nn.functional.relu(conv2_1)
        conv1_2 = nn.functional.conv2d(self.padding1(conv1_1), torch.cat((self.conv_share_weight_1, self.conv_ir_1), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_2 = nn.functional.conv2d(self.padding1(conv2_1), torch.cat((self.conv_share_weight_1, self.conv_vis_1), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_2 = nn.functional.relu(conv1_2)
        conv2_2 = nn.functional.relu(conv2_2)
        conv1_2 = torch.cat((conv1_2, conv1_1),dim=1)
        conv1_2 = torch.cat((conv1_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv1_1), dim=1)
        conv1_3 = nn.functional.conv2d(self.padding1(conv1_2), torch.cat((self.conv_share_weight_2, self.conv_ir_2), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_3 = nn.functional.conv2d(self.padding1(conv2_2), torch.cat((self.conv_share_weight_2, self.conv_vis_2), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_3 = nn.functional.relu(conv1_3)
        conv2_3 = nn.functional.relu(conv2_3)
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv1_3 = torch.cat((conv1_3, conv2_2), dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3), dim=1)
        conv2_3 = torch.cat((conv2_3, conv1_2), dim=1)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = nn.functional.relu(self.conv3_3(self.padding1(conv3_2)))
        conv3_4 = self.conv3_4(self.padding1(conv3_3))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_4
class Fusion_share_Net2(nn.Module):
    def __init__(self):
        super(Fusion_share_Net2, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))

        self.conv_ir_0 = nn.Conv2d(1, 16, 3,1,0)
        self.conv_vis_0 = nn.Conv2d(1, 16, 3,1,0)

        self.conv_share_weight_1 = nn.Parameter(torch.randn([16, 16, 3, 3],device=0))
        self.conv_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.conv_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))

        self.conv_share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))

        self.concat = nn.Sequential(Flgc2d(512, 256, 1, 1, 0, 1, groups=4, bias=False),nn.ReLU())

        self.conv3_1 = nn.Conv2d(256, 256, 3,1,0)
        self.conv3_2 = nn.Conv2d(256, 128, 3,1,0)
        self.conv3_3 = nn.Conv2d(128, 64, 3, 1,0)
        self.conv3_4 = nn.Conv2d(64, 1, 3, 1,0)
    def forward(self, infrared, visible):
        conv1_1 = self.conv_ir_0(self.padding1(infrared))
        conv2_1 = self.conv_vis_0(self.padding1(visible))
        conv1_1 = nn.functional.relu(conv1_1)
        conv2_1 = nn.functional.relu(conv2_1)
        conv1_2 = nn.functional.conv2d(self.padding1(conv1_1), torch.cat((self.conv_share_weight_1, self.conv_ir_1), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_2 = nn.functional.conv2d(self.padding1(conv2_1), torch.cat((self.conv_share_weight_1, self.conv_vis_1), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_2 = nn.functional.relu(conv1_2)
        conv2_2 = nn.functional.relu(conv2_2)
        conv1_2_s = nn.functional.conv2d(self.padding1(conv1_1),
                                       self.conv_ir_1,
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_2_s = nn.functional.conv2d(self.padding1(conv2_1),
                                       self.conv_vis_1,
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_2_s = nn.functional.relu(conv1_2_s)
        conv2_2_s = nn.functional.relu(conv2_2_s)
        self.featuremap1 = conv1_2_s.detach()
        self.featuremap2 = conv2_2_s.detach()
        conv1_2 = torch.cat((conv1_2, conv1_1),dim=1)
        conv1_2 = torch.cat((conv1_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv1_1), dim=1)
        conv1_3 = nn.functional.conv2d(self.padding1(conv1_2), torch.cat((self.conv_share_weight_2, self.conv_ir_2), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv2_3 = nn.functional.conv2d(self.padding1(conv2_2), torch.cat((self.conv_share_weight_2, self.conv_vis_2), dim=0),
                                       bias=None, stride=1, padding=0, dilation=1, groups=1)
        conv1_3 = nn.functional.relu(conv1_3)
        conv2_3 = nn.functional.relu(conv2_3)
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv1_3 = torch.cat((conv1_3, conv2_2), dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3), dim=1)
        conv2_3 = torch.cat((conv2_3, conv1_2), dim=1)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        concate_conv1 = self.concat(concate)
        coe_matrix_vis = torch.sigmoid(concate_conv1)
        concate = coe_matrix_vis*conv2_3+(1-coe_matrix_vis)*conv1_3
        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = nn.functional.relu(self.conv3_3(self.padding1(conv3_2)))
        conv3_4 = self.conv3_4(self.padding1(conv3_3))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_4

class Fusion_share_fft_Net2(nn.Module):
    def __init__(self):
        super(Fusion_share_fft_Net2, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))

        self.fft_ir_0 = nn.Sequential(nn.Conv2d(1,16,3,1,0),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.conv_ir_0 = nn.Sequential(FFTConv2d(16, 16, 1, 0, 1),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.fft_vis_0 = nn.Sequential(nn.Conv2d(1,16,3,1,0),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.conv_vis_0 = nn.Sequential(FFTConv2d(16, 16, 1, 0, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())

        self.conv_share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3)).cuda()
        self.conv_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3)).cuda()
        self.conv_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3)).cuda()
        self.bias_1 = nn.Parameter(torch.randn(32)).cuda()
        self.BN_1 = nn.BatchNorm2d(32)

        self.conv_share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3)).cuda()
        self.conv_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3)).cuda()
        self.conv_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3)).cuda()
        self.bias_2 = nn.Parameter(torch.randn(128)).cuda()
        self.BN_2 = nn.BatchNorm2d(128)

        self.concat = nn.Sequential(Flgc2d(512, 256, 1, 1, 0, 1, groups=4, bias=False),nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Conv2d(256, 256, 3,1,0),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(256, 128, 3,1,0),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(128, 64, 3,1,0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
        self.conv3_4 = nn.Conv2d(64, 1, 3, 1,0)
    def forward(self, infrared, visible):
        conv1_0 = self.fft_ir_0(self.padding1(infrared))
        conv2_0 = self.fft_vis_0(self.padding1(visible))
        conv1_1 = self.conv_ir_0(conv1_0)
        conv2_1 = self.conv_vis_0(conv2_0)
        conv1_2 = nn.functional.conv2d(self.padding1(conv1_1),
                                       torch.cat((self.conv_share_weight_1, self.conv_ir_1),dim=0),
                                       self.bias_1,
                                       1,0)
        conv2_2 = nn.functional.conv2d(self.padding1(conv2_1),
                                       torch.cat((self.conv_share_weight_1, self.conv_vis_1),dim=0),
                                       self.bias_1,
                                       1,0)
        conv1_2 = nn.functional.relu(self.BN_1(conv1_2))
        conv2_2 = nn.functional.relu(self.BN_1(conv2_2))
        conv1_2 = torch.cat((conv1_2, conv1_1),dim=1)
        conv1_2 = torch.cat((conv1_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv1_1), dim=1)
        conv1_3 = nn.functional.conv2d(self.padding1(conv1_2),
                                       torch.cat((self.conv_share_weight_2, self.conv_ir_2),dim=0),
                                       self.bias_2,
                                       1,0)
        conv2_3 = nn.functional.conv2d(self.padding1(conv2_2),
                                       torch.cat((self.conv_share_weight_2, self.conv_vis_2),dim=0),
                                       self.bias_2,
                                       1,0)
        conv1_3 = nn.functional.relu(self.BN_2(conv1_3))
        conv2_3 = nn.functional.relu(self.BN_2(conv2_3))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv1_3 = torch.cat((conv1_3, conv2_2), dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3), dim=1)
        conv2_3 = torch.cat((conv2_3, conv1_2), dim=1)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        concate_conv1 = self.concat(concate)
        coe_matrix_vis = torch.sigmoid(concate_conv1)
        concate = coe_matrix_vis*conv2_3+(1-coe_matrix_vis)*conv1_3
        conv3_1 = self.conv3_1(self.padding1(concate))
        conv3_2 = self.conv3_2(self.padding1(conv3_1))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        conv3_4 = self.conv3_4(self.padding1(conv3_3))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_4

class FusionFLGCNet(nn.Module):
    def __init__(self):
        super(FusionFLGCNet, self).__init__()
        self.padding1 = nn.ReplicationPad2d(1)
        self.conv1_1 = nn.Sequential(#nn.Conv2d(1, 32, 3, 1, 1),
                                     #nn.BatchNorm2d(1),
                                     #nn.ReLU(),
                                     Flgc2d(1, 32, 3, 1, 0, 1, groups=1, bias=False),
                                     #GroupNorm(32),
                                     nn.ReLU(),
                                     )
        self.conv1_2 = nn.Sequential(Flgc2d(32, 32, 3, 1, 0, 1, groups=4, bias=False),
                                     #GroupNorm(32),
                                     nn.ReLU(),
                                     )
        self.conv1_3 = nn.Sequential(Flgc2d(64, 64, 3, 1, 0, 1, groups=4, bias=False),
                                     #GroupNorm(64),
                                     nn.ReLU(),
                                     )
        self.conv2_1 = nn.Sequential(Flgc2d(1, 32, 3, 1, 0, 1, groups=1, bias=False),
                                     #GroupNorm(32),
                                     nn.ReLU(),
                                     )
        self.conv2_2 = nn.Sequential(Flgc2d(32, 32, 3, 1, 0, 1, groups=4, bias=False),
                                     #GroupNorm(32),
                                     nn.ReLU(),
                                     )
        self.conv2_3 = nn.Sequential(Flgc2d(64, 64, 3, 1, 0, 1, groups=4, bias=False),
                                     #GroupNorm(64),
                                     nn.ReLU(),
                                     )
        self.conv3_1 = nn.Sequential(Flgc2d(256, 128, 3, 1, 0, 1, groups=4, bias=False),
                                     #GroupNorm(128),
                                     nn.ReLU()
                                    )
        self.conv3_2 = nn.Sequential(#nn.Conv2d(128, 64, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     #nn.ReLU()
                                     Flgc2d(128,64,3,1,0,1,bias=False, groups=4),
                                     #GroupNorm(64),
                                     nn.ReLU()
                                     )
        self.conv3_3 = Flgc2d(64, 1, 3, 1,0, 1, bias=False, groups=4)
                                     #nn.BatchNorm2d(1),
                                     #nn.ReLU(),
                                     #Flgc2d(64,1,1,1,0,bias=False, groups=1)
        #self.conv3_4 = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv1_2 = self.conv1_2(self.padding1(conv1_1))
        conv1_2 = torch.cat([conv1_1, conv1_2], dim=1)
        conv1_3 = self.conv1_3(self.padding1(conv1_2))
        conv1_3 = torch.cat([conv1_3, conv1_2], dim=1)
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv2_2 = self.conv2_2(self.padding1(conv2_1))
        conv2_2 = torch.cat([conv2_1, conv2_2], dim=1)
        conv2_3 = self.conv2_3(self.padding1(conv2_2))
        conv2_3 = torch.cat([conv2_2, conv2_3], dim=1)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        conv3_1 = self.conv3_1(self.padding1(concate))
        conv3_2 = self.conv3_2(self.padding1(conv3_1))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_share_FLGC_Net2(nn.Module):
    def __init__(self):
        super(Fusion_share_FLGC_Net2, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))

        self.ir_0 = Flgc2d(1, 16, 3, 1, 0, 1, 1)
        self.vis_0 = Flgc2d(1, 16, 3, 1, 0, 1, 1)
        #self.BN_0 = nn.BatchNorm2d(16)

        self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)

        self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 32)
        self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 32)
        #self.BN_2 = nn.BatchNorm2d(128)

        self.concat = Flgc2d(512, 256, 1, 1, 0, 1, 32)
        self.conv3_1 = Flgc2d(256, 256, 3, 1, 0, 1, 32)
        self.conv3_2 = Flgc2d(256, 128, 3, 1, 0, 1, 32)
        self.conv3_3 = Flgc2d(128, 64, 3, 1, 0, 1, 16)
        self.conv3_4 = Flgc2d(64, 1, 3, 1, 0, 1, 16)
    def forward(self, infrared, visible):
        conv1_1 = self.ir_0(self.padding1(infrared))
        conv2_1 = self.vis_0(self.padding1(visible))
        conv1_1 = nn.functional.relu(conv1_1)
        conv2_1 = nn.functional.relu(conv2_1)
        conv1_2 = self.conv_ir_1(self.padding1(conv1_1))
        conv2_2 = self.conv_vis_1(self.padding1(conv2_1))
        conv1_2 = nn.functional.relu(conv1_2)
        conv2_2 = nn.functional.relu(conv2_2)
        conv1_2 = torch.cat((conv1_2, conv1_1),dim=1)
        conv1_2 = torch.cat((conv1_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv1_1), dim=1)
        conv1_3 = self.conv_ir_2(self.padding1(conv1_2))
        conv2_3 =self.conv_vis_2(self.padding1(conv2_2))
        conv1_3 = nn.functional.relu(conv1_3)
        conv2_3 = nn.functional.relu(conv2_3)
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv1_3 = torch.cat((conv1_3, conv2_2), dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3), dim=1)
        conv2_3 = torch.cat((conv2_3, conv1_2), dim=1)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        concate_conv1 = self.concat(concate)
        coe_matrix_vis = torch.sigmoid(concate_conv1)
        concate = coe_matrix_vis*conv2_3+(1-coe_matrix_vis)*conv1_3
        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = nn.functional.relu(self.conv3_3(self.padding1(conv3_2)))
        conv3_4 = self.conv3_4(self.padding1(conv3_3))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_4
class Fusion_share_NoFLGC_Net(nn.Module):
    def __init__(self):
        super(Fusion_share_NoFLGC_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))

        self.weight_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        self.weight_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.BN_0 = nn.BatchNorm2d(16)

        self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        #self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)

        self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.SConv_ir=SCConv(inplanes=256, planes=256, stride=1, padding=0, dilation=1, groups=16, pooling_r=1)
        self.SConv_vis = SCConv(inplanes=256, planes=256, stride=1, padding=0, dilation=1, groups=16, pooling_r=1)
        #self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 16)
        #self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 16)
        #self.BN_2 = nn.BatchNorm2d(128)

        self.concat = Flgc2d(1024, 512, 1, 1, 0, 1, 16)
        self.conv3_1 = Flgc2d(512, 256, 3, 1, 0, 1, 16)
        self.conv3_2 = Flgc2d(256, 128, 3, 1, 0, 1, 16)
        self.conv3_3 = Flgc2d(128, 64, 3, 1, 0, 1, 16)
        self.conv3_4 = Flgc2d(64, 1, 3, 1, 0, 1, 16)
    def forward(self, infrared, visible):
        conv1_1 = nn.functional.conv2d(self.padding1(infrared),self.weight_ir_0,None,1,0)
        conv2_1 = nn.functional.conv2d(self.padding1(visible),self.weight_vis_0,None,1,0)
        conv1_1 = nn.functional.relu(conv1_1)
        conv2_1 = nn.functional.relu(conv2_1)
        conv1_2 = nn.functional.conv2d(self.padding1(conv1_1),
                                       torch.cat((self.share_weight_1, self.weight_ir_1),dim=0),
                                       None,
                                       1,0)
        conv2_2 = nn.functional.conv2d(self.padding1(conv2_1),
                                       torch.cat((self.share_weight_1, self.weight_vis_1),dim=0),
                                       None,
                                       1,0)
        conv1_2 = nn.functional.relu(conv1_2)
        conv2_2 = nn.functional.relu(conv2_2)
        conv1_2 = torch.cat((conv1_2, conv1_1),dim=1)
        conv1_2 = torch.cat((conv1_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1), dim=1)
        conv2_2 = torch.cat((conv2_2, conv1_1), dim=1)
        conv1_3 = nn.functional.conv2d(self.padding1(conv1_2),
                                       torch.cat((self.share_weight_2, self.weight_ir_2),dim=0),
                                       None,
                                       1,0)
        conv2_3 =nn.functional.conv2d(self.padding1(conv2_2),
                                       torch.cat((self.share_weight_2, self.weight_ir_2),dim=0),
                                       None,
                                       1,0)
        conv1_3 = nn.functional.relu(conv1_3)
        conv2_3 = nn.functional.relu(conv2_3)
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv1_3 = torch.cat((conv1_3, conv2_2), dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3), dim=1)
        conv2_3 = torch.cat((conv2_3, conv1_2), dim=1)
        SConv_ir=self.SConv_ir(conv1_3)
        SConv_vis = self.SConv_vis(conv2_3)
        concate = torch.cat((SConv_ir, SConv_vis), dim=1)
        concate_conv1 = self.concat(concate)
        coe_matrix_vis = torch.sigmoid(concate_conv1)
        concate = coe_matrix_vis*SConv_vis+(1-coe_matrix_vis)*SConv_ir
        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = nn.functional.relu(self.conv3_3(self.padding1(conv3_2)))
        conv3_4 = self.conv3_4(self.padding1(conv3_3))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_4
class Fusion_Noshare_NoFLGC_Net(nn.Module):
    def __init__(self):
        super(Fusion_Noshare_NoFLGC_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.padding3 = nn.ReflectionPad2d((1, 1, 1, 1))

        #self.weight_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.weight_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.BN_0 = nn.BatchNorm2d(16)

        #self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        #self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)

        #self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        #self.SConv_ir=SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 16)
        #self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 16)
        #self.BN_2 = nn.BatchNorm2d(128)

        #self.concat = nn.Conv2d(128, 64, 1,1,0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        #SConv_ir=self.SConv_ir(conv1_3)
        #SConv_vis = self.SConv_vis(conv2_3)
        #concate = torch.cat((conv1_3, conv2_3), dim=1)
        #concate_conv1 = self.concat(concate)
        #coe_matrix_vis = torch.sigmoid(concate_conv1)
        #concate = coe_matrix_vis*conv2_3+(1-coe_matrix_vis)*conv1_3
        concate=conv1_3+conv2_3
        conv3_1 = self.conv3_1(self.padding1(concate))
        conv3_2 = self.conv3_2(self.padding1(conv3_1))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_HDC_DTCWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_HDC_DTCWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))

        #self.weight_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.weight_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.BN_0 = nn.BatchNorm2d(16)

        #self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        #self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)

        #self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.dwt_1 = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.dwt_2 = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        #self.dwt_1=DWT_2D('haar')
        #self.dwt_2 = DWT_2D('haar')
        #self.idwt = IDWT_2D('haar')
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 16)
        #self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 16)
        #self.BN_2 = nn.BatchNorm2d(128)

        # self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        # self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        # self.concat_hl = nn.Conv2d(128, 64, 1,1,0)
        # self.concat_hh = nn.Conv2d(128, 64, 1,1,0)
        self.concat_l = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_h = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        #dwt_ll_1,dwt_lh_1,dwt_hl_1,dwt_hh_1=self.dwt_1(conv1_3)
        #dwt_ll_2,dwt_lh_2,dwt_hl_2,dwt_hh_2 = self.dwt_2(conv2_3)
        dwt_l_1,dwt_h_1=self.dwt_1(conv1_3)
        dwt_l_2,dwt_h_2= self.dwt_2(conv2_3)
        dtcwt_h_1_1 = dwt_h_1[0]
        dtcwt_h_1_2 = dwt_h_1[1]
        dtcwt_h_1_3 = dwt_h_1[2]
        self.featuremap1 = torch.mean(conv2_3, dim=1)
        dwt_l_1 = self.SConv_ir(dwt_l_1)
        dwt_l_2 = self.SConv_ir(dwt_l_2)
        concate_ll = torch.cat((dwt_l_1, dwt_l_2), dim=1)
        concate_conv_ll = self.concat_l(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll * dwt_l_2 + (1 - coe_matrix_vis_ll) * dwt_l_1
        concate_hh=[]
        concate_hh_0=dwt_h_1[0]+dwt_h_2[0]
        concate_hh_1 = dwt_h_1[1] + dwt_h_2[1]
        concate_hh_2 = dwt_h_1[2] + dwt_h_2[2]
        concate_hh.append(concate_hh_0)
        concate_hh.append(concate_hh_1)
        concate_hh.append(concate_hh_2)
        concate = self.idwt((concate_ll, concate_hh))
        # dwt_ll_1 = self.SConv_ir(dwt_ll_1)
        # dwt_ll_2 = self.SConv_vis(dwt_ll_2)
        # concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        # concate_conv_ll = self.concat_ll(concate_ll)
        # coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        # concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1
        #
        # concate_lh = torch.cat((dwt_lh_1, dwt_lh_2), dim=1)
        # concate_conv_lh = self.concat_lh(concate_lh)
        # coe_matrix_vis_lh = torch.sigmoid(concate_conv_lh)
        # concate_lh = coe_matrix_vis_lh * dwt_lh_2 + (1 - coe_matrix_vis_lh) * dwt_lh_1
        #
        # concate_hl = torch.cat((dwt_hl_1, dwt_hl_2), dim=1)
        # concate_conv_hl = self.concat_hl(concate_hl)
        # coe_matrix_vis_hl = torch.sigmoid(concate_conv_hl)
        # concate_hl = coe_matrix_vis_hl * dwt_hl_2 + (1 - coe_matrix_vis_hl) * dwt_hl_1
        #
        # concate_hh = torch.cat((dwt_hh_1, dwt_hh_2), dim=1)
        # concate_conv_hh = self.concat_hh(concate_hh)
        # coe_matrix_vis_hh = torch.sigmoid(concate_conv_hh)
        # concate_hh = coe_matrix_vis_hh * dwt_hh_2 + (1 - coe_matrix_vis_hh) * dwt_hh_1
        #
        # concate = self.idwt(concate_ll,concate_lh,concate_hl,concate_hh)

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_HDC_DWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_HDC_DWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.dwt_1=DWT_2D('haar')
        self.dwt_2 = DWT_2D('haar')
        self.idwt = IDWT_2D('haar')
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)

        self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hl = nn.Conv2d(128, 64, 1,1,0)
        self.concat_hh = nn.Conv2d(128, 64, 1,1,0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        self.featuremap1 = torch.mean(conv2_1, dim=1)
        dwt_ll_1,dwt_lh_1,dwt_hl_1,dwt_hh_1=self.dwt_1(conv1_3)
        dwt_ll_2,dwt_lh_2,dwt_hl_2,dwt_hh_2 = self.dwt_2(conv2_3)
        dwt_ll_1 = self.SConv_ir(dwt_ll_1)
        dwt_ll_2 = self.SConv_vis(dwt_ll_2)
        concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        concate_conv_ll = self.concat_ll(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1

        concate_lh = torch.cat((dwt_lh_1, dwt_lh_2), dim=1)
        concate_conv_lh = self.concat_lh(concate_lh)
        coe_matrix_vis_lh = torch.sigmoid(concate_conv_lh)
        concate_lh = coe_matrix_vis_lh * dwt_lh_2 + (1 - coe_matrix_vis_lh) * dwt_lh_1

        concate_hl = torch.cat((dwt_hl_1, dwt_hl_2), dim=1)
        concate_conv_hl = self.concat_hl(concate_hl)
        coe_matrix_vis_hl = torch.sigmoid(concate_conv_hl)
        concate_hl = coe_matrix_vis_hl * dwt_hl_2 + (1 - coe_matrix_vis_hl) * dwt_hl_1

        concate_hh = torch.cat((dwt_hh_1, dwt_hh_2), dim=1)
        concate_conv_hh = self.concat_hh(concate_hh)
        coe_matrix_vis_hh = torch.sigmoid(concate_conv_hh)
        concate_hh = coe_matrix_vis_hh * dwt_hh_2 + (1 - coe_matrix_vis_hh) * dwt_hh_1

        concate = self.idwt(concate_ll,concate_lh,concate_hl,concate_hh)

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_DWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_DWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))

        #self.weight_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.weight_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.BN_0 = nn.BatchNorm2d(16)

        #self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        #self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)
        #self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.dwt_1 = DWT_2D('haar')
        self.dwt_2 = DWT_2D('haar')
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())

        self.idwt = IDWT_2D('haar')
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 16)
        #self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 16)
        #self.BN_2 = nn.BatchNorm2d(128)

        self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hl = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hh = nn.Conv2d(128, 64, 1, 1, 0)

        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        L1,H1,dwt_ll_1, dwt_lh_1, dwt_hl_1, dwt_hh_1 = self.dwt_1(infrared)
        L,H,dwt_ll_2, dwt_lh_2, dwt_hl_2, dwt_hh_2 = self.dwt_2(visible)
        self.featuremap1 = torch.sum(H, dim=1)
        conv1_1 = self.conv1_1(self.padding1(dwt_ll_1))
        conv2_1 = self.conv2_1(self.padding1(dwt_ll_2))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)

        dwt_ll_1 = self.SConv_ir(conv1_3)
        dwt_ll_2 = self.SConv_vis(conv2_3)
        concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        concate_conv_ll = self.concat_ll(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1

        conv1_1lh = self.conv1_1(self.padding1(dwt_lh_1))
        conv2_1lh = self.conv2_1(self.padding1(dwt_lh_2))
        conv1_2lh= self.conv1_2(self.padding2(conv1_1lh))
        conv2_2lh = self.conv2_2(self.padding2(conv2_1lh))
        conv1_2lh = torch.cat((conv1_1lh, conv1_2lh), dim=1)
        conv2_2lh = torch.cat((conv2_1lh, conv2_2lh), dim=1)
        conv1_3lh = self.conv1_3(self.padding3(conv1_2lh))
        conv2_3lh = self.conv2_3(self.padding3(conv2_2lh))
        conv1_3lh = torch.cat((conv1_2lh, conv1_3lh), dim=1)
        conv2_3lh = torch.cat((conv2_2lh, conv2_3lh), dim=1)
        dwt_lh_1 = self.SConv_ir(conv1_3lh)
        dwt_lh_2 = self.SConv_vis(conv2_3lh)
        concate_lh = torch.cat((dwt_lh_1, dwt_lh_2), dim=1)
        concate_conv_lh = self.concat_lh(concate_lh)
        coe_matrix_vis_lh = torch.sigmoid(concate_conv_lh)
        concate_lh = coe_matrix_vis_lh * dwt_lh_2 + (1 - coe_matrix_vis_lh) * dwt_lh_1

        conv1_1hl = self.conv1_1(self.padding1(dwt_hl_1))
        conv2_1hl = self.conv2_1(self.padding1(dwt_hl_2))
        conv1_2hl = self.conv1_2(self.padding2(conv1_1hl))
        conv2_2hl = self.conv2_2(self.padding2(conv2_1hl))
        conv1_2hl = torch.cat((conv1_1hl, conv1_2hl), dim=1)
        conv2_2hl = torch.cat((conv2_1hl, conv2_2hl), dim=1)
        conv1_3hl = self.conv1_3(self.padding3(conv1_2hl))
        conv2_3hl = self.conv2_3(self.padding3(conv2_2hl))
        conv1_3hl = torch.cat((conv1_2hl, conv1_3hl), dim=1)
        conv2_3hl = torch.cat((conv2_2hl, conv2_3hl), dim=1)
        dwt_hl_1 = self.SConv_ir(conv1_3hl)
        dwt_hl_2 = self.SConv_vis(conv2_3hl)
        concate_hl = torch.cat((dwt_hl_1, dwt_hl_2), dim=1)
        concate_conv_hl = self.concat_hl(concate_hl)
        coe_matrix_vis_hl = torch.sigmoid(concate_conv_hl)
        concate_hl = coe_matrix_vis_hl * dwt_hl_2 + (1 - coe_matrix_vis_hl) * dwt_hl_1

        conv1_1hh = self.conv1_1(self.padding1(dwt_hh_1))
        conv2_1hh = self.conv2_1(self.padding1(dwt_hh_2))
        conv1_2hh = self.conv1_2(self.padding2(conv1_1hh))
        conv2_2hh = self.conv2_2(self.padding2(conv2_1hh))
        conv1_2hh = torch.cat((conv1_1hh, conv1_2hh), dim=1)
        conv2_2hh = torch.cat((conv2_1hh, conv2_2hh), dim=1)
        conv1_3hh = self.conv1_3(self.padding3(conv1_2hh))
        conv2_3hh = self.conv2_3(self.padding3(conv2_2hh))
        conv1_3hh = torch.cat((conv1_2hh, conv1_3hh), dim=1)
        conv2_3hh = torch.cat((conv2_2hh, conv2_3hh), dim=1)
        dwt_hh_1 = self.SConv_ir(conv1_3hh)
        dwt_hh_2 = self.SConv_vis(conv2_3hh)
        concate_hh = torch.cat((dwt_hh_1, dwt_hh_2), dim=1)
        concate_conv_hh = self.concat_hh(concate_hh)
        coe_matrix_vis_hh = torch.sigmoid(concate_conv_hh)
        concate_hh = coe_matrix_vis_hh * dwt_hh_2 + (1 - coe_matrix_vis_hh) * dwt_hh_1
        concate = self.idwt(concate_ll, concate_lh, concate_hl, concate_hh)

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #concate = self.idwt(conv3_3, concate_lh, concate_hl, concate_hh)
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_NOHDC_DWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_NOHDC_DWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        self.dwt_1=DWT_2D('haar')
        self.dwt_2 = DWT_2D('haar')
        self.idwt = IDWT_2D('haar')
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)

        self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hl = nn.Conv2d(128, 64, 1,1,0)
        self.concat_hh = nn.Conv2d(128, 64, 1,1,0)
        # self.concat_l = nn.Conv2d(128, 64, 1, 1, 0)
        # self.concat_h = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding1(conv1_1))
        conv2_2 = self.conv2_2(self.padding1(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding1(conv1_2))
        conv2_3 = self.conv2_3(self.padding1(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        self.featuremap1 = torch.mean(conv2_1, dim=1)
        dwt_ll_1,dwt_lh_1,dwt_hl_1,dwt_hh_1=self.dwt_1(conv1_3)
        dwt_ll_2,dwt_lh_2,dwt_hl_2,dwt_hh_2 = self.dwt_2(conv2_3)
        dwt_ll_1 = self.SConv_ir(dwt_ll_1)
        dwt_ll_2 = self.SConv_vis(dwt_ll_2)
        concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        concate_conv_ll = self.concat_ll(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1

        concate_lh = torch.cat((dwt_lh_1, dwt_lh_2), dim=1)
        concate_conv_lh = self.concat_lh(concate_lh)
        coe_matrix_vis_lh = torch.sigmoid(concate_conv_lh)
        concate_lh = coe_matrix_vis_lh * dwt_lh_2 + (1 - coe_matrix_vis_lh) * dwt_lh_1

        concate_hl = torch.cat((dwt_hl_1, dwt_hl_2), dim=1)
        concate_conv_hl = self.concat_hl(concate_hl)
        coe_matrix_vis_hl = torch.sigmoid(concate_conv_hl)
        concate_hl = coe_matrix_vis_hl * dwt_hl_2 + (1 - coe_matrix_vis_hl) * dwt_hl_1

        concate_hh = torch.cat((dwt_hh_1, dwt_hh_2), dim=1)
        concate_conv_hh = self.concat_hh(concate_hh)
        coe_matrix_vis_hh = torch.sigmoid(concate_conv_hh)
        concate_hh = coe_matrix_vis_hh * dwt_hh_2 + (1 - coe_matrix_vis_hh) * dwt_hh_1

        concate = self.idwt(concate_ll,concate_lh,concate_hl,concate_hh)

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_HDC_NODWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_HDC_NODWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)

        self.concat =nn.Conv2d(128, 64, 1,1,0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        conv1_3 = self.SConv_ir(conv1_3)
        conv2_3 = self.SConv_vis(conv2_3)
        concate = torch.cat((conv1_3, conv2_3), dim=1)
        concate_conv_ll = self.concat(concate)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*conv1_3+(1-coe_matrix_vis_ll)*conv2_3
        conv3_1 = self.conv3_1(self.padding1(concate_ll))
        conv3_2 = self.conv3_2(self.padding1(conv3_1))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_HDC_DWT_NOSCNet(nn.Module):
    def __init__(self):
        super(Fusion_HDC_DWT_NOSCNet, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.dwt_1=DWT_2D('haar')
        self.dwt_2 = DWT_2D('haar')
        self.idwt = IDWT_2D('haar')
        #self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)

        self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hl = nn.Conv2d(128, 64, 1,1,0)
        self.concat_hh = nn.Conv2d(128, 64, 1,1,0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        self.featuremap1 = torch.mean(conv2_1, dim=1)
        dwt_ll_1,dwt_lh_1,dwt_hl_1,dwt_hh_1=self.dwt_1(conv1_3)
        dwt_ll_2,dwt_lh_2,dwt_hl_2,dwt_hh_2 = self.dwt_2(conv2_3)
        #dwt_ll_1 = self.SConv_ir(dwt_ll_1)
        #dwt_ll_2 = self.SConv_vis(dwt_ll_2)
        concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        concate_conv_ll = self.concat_ll(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1

        concate_lh = torch.cat((dwt_lh_1, dwt_lh_2), dim=1)
        concate_conv_lh = self.concat_lh(concate_lh)
        coe_matrix_vis_lh = torch.sigmoid(concate_conv_lh)
        concate_lh = coe_matrix_vis_lh * dwt_lh_2 + (1 - coe_matrix_vis_lh) * dwt_lh_1

        concate_hl = torch.cat((dwt_hl_1, dwt_hl_2), dim=1)
        concate_conv_hl = self.concat_hl(concate_hl)
        coe_matrix_vis_hl = torch.sigmoid(concate_conv_hl)
        concate_hl = coe_matrix_vis_hl * dwt_hl_2 + (1 - coe_matrix_vis_hl) * dwt_hl_1

        concate_hh = torch.cat((dwt_hh_1, dwt_hh_2), dim=1)
        concate_conv_hh = self.concat_hh(concate_hh)
        coe_matrix_vis_hh = torch.sigmoid(concate_conv_hh)
        concate_hh = coe_matrix_vis_hh * dwt_hh_2 + (1 - coe_matrix_vis_hh) * dwt_hh_1

        concate = self.idwt(concate_ll,concate_lh,concate_hl,concate_hh)

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_NOHDC_NODWT_NOSCNet(nn.Module):
    def __init__(self):
        super(Fusion_NOHDC_NODWT_NOSCNet, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,1,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,1,8),
                                     nn.ReLU())
        self.concat = nn.Conv2d(128, 64, 1,1,0)
        # self.concat_l = nn.Conv2d(128, 64, 1, 1, 0)
        # self.concat_h = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(self.padding1(infrared))
        conv2_1 = self.conv2_1(self.padding1(visible))
        conv1_2 = self.conv1_2(self.padding1(conv1_1))
        conv2_2 = self.conv2_2(self.padding1(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding1(conv1_2))
        conv2_3 = self.conv2_3(self.padding1(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)

        concate = torch.cat((conv1_3, conv2_3), dim=1)
        concate_conv_ll = self.concat(concate)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*conv1_3+(1-coe_matrix_vis_ll)*conv2_3

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate_ll)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_DTCWT_Net(nn.Module):
    def __init__(self):
        super(Fusion_DTCWT_Net, self).__init__()
        self.padding1 = nn.ReflectionPad2d((1,1,1,1))
        self.padding2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.padding3 = nn.ReflectionPad2d((3, 3, 3, 3))

        #self.weight_ir_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.weight_vis_0 = nn.Parameter(torch.randn(16, 1, 3, 3))
        #self.BN_0 = nn.BatchNorm2d(16)

        #self.share_weight_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_ir_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.weight_vis_1 = nn.Parameter(torch.randn(16, 16, 3, 3))
        #self.conv_ir_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_ir_1), dim=0), 1, 0, 1, 16)
        #self.conv_vis_1 = Flgc2d_share(torch.cat((self.share_weight_1, self.weight_vis_1), dim=0), 1, 0, 1, 16)
        #self.BN_1 = nn.BatchNorm2d(32)
        #self.share_weight_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_ir_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        #self.weight_vis_2 = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.dwt_1 = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.dwt_2 = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3,1,0,1,1),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3,1,0,2,8),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 32, 3,1,0,3,8),
                                     nn.ReLU())
        self.SConv_ir = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        self.SConv_vis = SCConv(inplanes=64, planes=64, stride=1, padding=0, dilation=1, groups=16, pooling_r=2)
        #self.conv_ir_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_ir_2), dim=0), 1, 0, 1, 16)
        #self.conv_vis_2 = Flgc2d_share(torch.cat((self.share_weight_2, self.weight_vis_2), dim=0), 1, 0, 1, 16)
        #self.BN_2 = nn.BatchNorm2d(128)

        self.concat_ll =nn.Conv2d(128, 64, 1,1,0)
        self.concat_lh = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hl = nn.Conv2d(128, 64, 1, 1, 0)
        self.concat_hh = nn.Conv2d(128, 64, 1, 1, 0)

        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, 3,1,0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3,1,0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 1, 3,1,0),
                                     nn.ReLU())
    def forward(self, infrared, visible):
        dtcwt_l_1, dtcwt_h_1 = self.dwt_1(infrared)
        dtcwt_l_2, dtcwt_h_2 = self.dwt_2(visible)
        conv1_1 = self.conv1_1(self.padding1(dtcwt_l_1))
        conv2_1 = self.conv2_1(self.padding1(dtcwt_l_2))
        conv1_2 = self.conv1_2(self.padding2(conv1_1))
        conv2_2 = self.conv2_2(self.padding2(conv2_1))
        conv1_2 = torch.cat((conv1_1, conv1_2),dim=1)
        conv2_2 = torch.cat((conv2_2, conv2_1),dim=1)
        conv1_3 = self.conv1_3(self.padding3(conv1_2))
        conv2_3 = self.conv2_3(self.padding3(conv2_2))
        conv1_3 = torch.cat((conv1_2, conv1_3),dim=1)
        conv2_3 = torch.cat((conv2_2, conv2_3),dim=1)
        dwt_ll_1 = self.SConv_ir(conv1_3)
        dwt_ll_2 = self.SConv_vis(conv2_3)
        concate_ll = torch.cat((dwt_ll_1, dwt_ll_2), dim=1)
        concate_conv_ll = self.concat_ll(concate_ll)
        coe_matrix_vis_ll = torch.sigmoid(concate_conv_ll)
        concate_ll = coe_matrix_vis_ll*dwt_ll_2+(1-coe_matrix_vis_ll)*dwt_ll_1
        self.featuremap1 = torch.sum(dwt_ll_2, dim=1)
        concate_hh = []
        concate_hh_0 = dtcwt_h_1[0] + dtcwt_h_2[0]
        concate_hh_1 = dtcwt_h_1[1] + dtcwt_h_2[1]
        concate_hh_2 = dtcwt_h_1[2] + dtcwt_h_2[2]
        concate_hh.append(concate_hh_0)
        concate_hh.append(concate_hh_1)
        concate_hh.append(concate_hh_2)
        concate = self.idwt((concate_ll, concate_hh))

        conv3_1 = nn.functional.relu(self.conv3_1(self.padding1(concate)))
        conv3_2 = nn.functional.relu(self.conv3_2(self.padding1(conv3_1)))
        conv3_3 = self.conv3_3(self.padding1(conv3_2))
        #concate = self.idwt(conv3_3, concate_lh, concate_hl, concate_hh)
        #conv3_4 = self.conv3_4(conv3_3)
        return conv3_3
class Fusion_swin_transformer_net(nn.Module):
    def __init__(self):
        super(Fusion_swin_transformer_net, self).__init__()
        self.V_swin_trans = SwinTransformer(img_size=224, patch_size=4, in_chans=1, num_classes=512, embed_dim=128,
                                            depths=[2], num_heads=[8], window_size=7, mlp_ratio=4.,
                                            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                                            drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                                            use_checkpoint=False)
        self.T_swin_trans = SwinTransformer(img_size=224, patch_size=4, in_chans=1, num_classes=512, embed_dim=64,
                                            depths=[2], num_heads=[8], window_size=7, mlp_ratio=4.,
                                            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                                            drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                                            use_checkpoint=False)
        self.mlp = Mlp(in_features=128, hidden_features=256,out_features=1, act_layer=nn.ReLU, drop=0.)
    def forward(self, infrared, visible):
        V_swin_trans = nn.functional.relu(self.V_swin_trans(visible))
        #T_swin_trans = nn.functional.relu(self.T_swin_trans(infrared))
        #concat = torch.cat([V_swin_trans,T_swin_trans],dim=1)
        B,C,H,W=visible.shape
        #concat = V_swin_trans.view(B,H,W,C)
        x = V_swin_trans.view(B, 128, 56, 56)
        self.featuremap1 = x.detach()
        x = F.interpolate(x, visible.size()[2:])
        self.featuremap2 = x.detach()
        x=x.view(B,224*224,128)
        conv2_1=self.mlp(x)
        conv2_1 = conv2_1.view(B,-1,H,W)
        #conv2_2 = torch.tanh(conv2_1)
        return conv2_1

