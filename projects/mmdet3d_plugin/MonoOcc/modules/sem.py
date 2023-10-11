import torch.nn as nn
class Sem_Decoder(nn.Module):
    def __init__(self, input_channel, num_classes, ratio):
        super().__init__() 
        if ratio == 4:
            self.kernel_size = 3
            self.dilation = 1
            self.output_padding = 1
        elif ratio == 2:
            self.kernel_size = 1
            self.dilation = 1
            self.output_padding = 1
        self.upconv = SeparableDeConv2d(input_channel,input_channel,
                                      stride=ratio, kernel_size=self.kernel_size, 
                                      dilation=self.dilation, output_padding=self.output_padding)
        self.upconv2 = SeparableDeConv2d(input_channel,input_channel,
                                      stride=ratio, kernel_size=self.kernel_size, 
                                      dilation=self.dilation, output_padding=self.output_padding)
        self.head = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(input_channel, num_classes, kernel_size=1),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        B = x.shape[0]
        for bs in range(B):
            x = x[bs]
            x = self.upconv(x)
            x = self.upconv2(x)      
            for i, layer in enumerate(self.head):
                x = layer(x)
        x = x.unsqueeze(0)
        return x

class SeparableDeConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=4,padding=0,dilation=4,bias=False,output_padding=0):
        super(SeparableDeConv2d,self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,padding=padding,dilation=dilation,
            output_padding=output_padding)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn = nn.Sequential(          
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False)
        )
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x
    
class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x