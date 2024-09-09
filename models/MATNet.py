import torch
import torch.nn as nn
import torch.nn.functional as F
# from pvtv2 import pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class MTANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, training = True, channel=32):
        self.training = training
        super(MTANet, self).__init__()
        # ---- ResNet Backbone ----
        # self.backbone = pvt_v2_b2()
        # path = 'lib/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        # ---- Partial Decoder ----
        self.agg = aggregation(channel)
        # ---- deconvolution 4 ----
        self.de4_dconv = BasicConv2d(512, 320, kernel_size=3, padding=1)
        self.de4_conv1 = BasicConv2d(320, 320, kernel_size=3, padding=1)
        self.de4_conv2 = BasicConv2d(320, 320, kernel_size=3, padding=1)
        self.de4_conv3 = BasicConv2d(320, 1, kernel_size=3, padding=1)

        self.de3_dconv = BasicConv2d(320, 128, kernel_size=3, padding=1)
        self.de3_conv1 = BasicConv2d(128, 128, kernel_size=3, padding=1)
        self.de3_conv2 = BasicConv2d(128, 128, kernel_size=3, padding=1)
        self.de3_conv3 = BasicConv2d(128, 1, kernel_size=3, padding=1)
        self.de2_dconv = BasicConv2d(128, 64, kernel_size=3, padding=1)
        self.de2_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.de2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.de2_conv3 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(512)

        self.fc = nn.Linear(1024, 8)
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 40)
        self.fc2 = nn.Linear(8, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(10, 20)
        self.fc_out1 = nn.Linear(15, 2)


    def forward(self, data=None, CT_data= None):
        x = torch.squeeze(data, dim=0)
        y = torch.squeeze(CT_data, dim=0)
        x = self.fc0(x)
        y = self.fc(y)

        # lateral_map_1 = self.avgpool(x)
        lateral_map_1 = torch.unsqueeze(torch.mean(x, dim=0), dim=0)
        # lateral_map_1 = lateral_map_1.view(lateral_map_1.size(0), -1)


        ######
        lateral_map_out_1 = self.fc1(lateral_map_1) #512--40
        lateral_map_1 = lateral_map_out_1[:,:35]
        bottleneck_lateral_map_1 = lateral_map_out_1[:,35:]


        clinical_map_out_1 = self.fc2(y.float()) #8--20
        clinical_map_1 = clinical_map_out_1[:,:15]
        bottleneck_clinical_map_1 = clinical_map_out_1[:,15:]

        bottleneck_out_1 = (bottleneck_lateral_map_1 + bottleneck_clinical_map_1)/2
        lateral_map_out_2 = self.fc3(torch.cat([lateral_map_1[:, 20:], bottleneck_out_1], dim=1)) #20--20

        lateral_map_2 = lateral_map_out_2[:,:15]
        bottleneck_lateral_map_2 = lateral_map_out_2[:,15:]

        clinical_map_out_2 = self.fc4(torch.cat([clinical_map_1[:, 10:], bottleneck_out_1], dim=1)) #10--20
        clinical_map_2 = clinical_map_out_2[:,:15]
        bottleneck_clinical_map_2 = clinical_map_out_2[:,15:]

        bottleneck_out_2 = (bottleneck_lateral_map_2 + bottleneck_clinical_map_2) / 2
        out = torch.cat([lateral_map_2[:, 10:], bottleneck_out_2], dim=1)
        logits = self.fc_out1(torch.cat([out, clinical_map_2[:, 10:]], dim=1))
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print(Y_hat.shape)
        Y_prob = F.softmax(logits, dim=1)
        ###

        return logits, Y_prob, Y_hat


