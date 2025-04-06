import torch.nn as nn



class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                   padding=1),
                                nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                             padding=1),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                             padding=1),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                             padding=1),
                                   nn.LeakyReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                             padding=1),
                                   nn.LeakyReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                             padding=1),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, input):

        x=self.conv1(input)
        x=self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)

        return out


class basemodel(nn.Module):
    def __init__(self):
        super(basemodel, self).__init__()


        self.re_blocks = BaseModule()
        self.conv = nn.Conv2d(64, 12, 3, 1,1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, input):
        res = nn.functional.interpolate(input, scale_factor=2, mode='bicubic', align_corners=False)
        x=self.re_blocks(input)#提取特征
        x=self.conv(x)#缩放通道
        output=self.pixel_shuffle(x)#上采样超分重建
        output=output+res#采用全局残差学习的策略，将经过双三次（Bicubic）插值后的目标帧与重建网络的输出相加
        return output
