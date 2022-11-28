#inpired from https://github.com/zsef123/EfficientNets-PyTorch
import math 
import torch
from torch import nn, optim
import torch.nn.functional as F


#Squeeze and extraction
class sqExc(nn.Module):
    def __init__(self, in_f, out_f):
        super(sqExc, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_f, out_f, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(out_f, in_f, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ = self.se(x)
        x = x*x_
        return x

#it does not reduce the size of the image, It preserves the size but modifies the channels this efficientnet 
#more extracted information to work with. 
class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])
        
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

        return out

#Batch-Normalization Activation 
def convBNact(in_f, out_f, kernel_size,
                stride=1, bias=True,
                eps=1e-3, momentum=0.01):
    out = nn.Sequential(
        SamePadConv2d(in_f, out_f, kernel_size, stride, bias=bias),
        nn.BatchNorm2d(out_f, eps, momentum),
        nn.SiLU()
    )
    return out

#MobileNet
class mbConv(nn.Module):
    def __init__(self, in_f=3, out_f=1, expand=1,
                 kernel_size=3, stride=1, skip=True,
                 se_ratio=0.25, dc_ratio=0.2):
        super(mbConv, self).__init__()
        hid_f = in_f * expand
        
        #expanding the network
        self.expand = convBNact(in_f, hid_f, kernel_size=1, bias=False) if expand != 1 else nn.Identity()
        
        #increasing depth to extract information
        self.depthwise = convBNact(hid_f, hid_f,kernel_size=kernel_size, stride=stride, bias=False)
        
        #squeeze and extraction
        self.s_e = sqExc(hid_f, int(in_f * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(hid_f, out_f, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_f, 1e-3, 0.01)
        )

        self.skip = skip and (stride == 1) and (in_f == out_f)

        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        x = self.expand(inputs)
        x = self.depthwise(x)
        x = self.s_e(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x

class mb_Block(nn.Module):
    def __init__(self, in_f, out_f, expand, kernel, stride, no_of_layers, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [mbConv(in_f, out_f, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, no_of_layers):
            layers.append(mbConv(out_f, out_f, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.2, drop_connect_rate=0.2, num_classes=2):
        super(EfficientNet, self).__init__()
        self.stem = convBNact(3, 32, kernel_size=3, stride=2, bias=False)
        
        self.blocks = nn.Sequential(
            mb_Block(32, 16, 1, 3, 1, 1, True, 0.25, drop_connect_rate),
            mb_Block(16, 24, 6, 3, 2, 2, True, 0.25, drop_connect_rate),
            mb_Block(24, 40, 6, 5, 2, 2, True, 0.25, drop_connect_rate),
            mb_Block(40, 80, 6, 3, 2, 3, True, 0.25, drop_connect_rate),
            mb_Block(80, 112, 6, 5, 1, 3, True, 0.25, drop_connect_rate),
            mb_Block(112, 192, 6, 5, 2, 4, True, 0.25, drop_connect_rate),
            mb_Block(192, 320, 6, 3, 1, 1, True, 0.25, drop_connect_rate)
        )

        self.head = nn.Sequential(
            *convBNact(320, 1280, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        out = self.head(x)
        return out

if __name__ == "__main__":
    X = torch.randn(10, 3, 32, 32)
    print(EfficientNet()(X))