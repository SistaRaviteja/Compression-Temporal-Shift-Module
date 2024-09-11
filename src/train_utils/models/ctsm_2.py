import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class learnTSM(nn.Module):
    def __init__(self, in_channels, version='zero', alpha=1, inplace=True):
        super(learnTSM, self).__init__()
        self.alpha = alpha
        if self.alpha > 1:
            self.split_size = in_channels//self.alpha
        else:
            self.split_size = in_channels
        self.pre_conv = nn.Conv2d(self.split_size, self.split_size//4, kernel_size=3, stride=1, padding=1)
        self.post_conv = nn.Conv2d(self.split_size, self.split_size//4, kernel_size=3, stride=1, padding=1)
        self.main_conv = nn.Conv2d(self.split_size, self.split_size//2, kernel_size=3, stride=1, padding=1)
        self.version = version
        self.inplace = inplace

    def forward(self, tensor, tsm_length=3):
        shape = T, C, H, W = tensor.shape
        split_size = self.split_size
        shift_tensor, main_tensor = tensor.split([split_size, C - split_size], dim=1)
        main_conv_tensor = self.main_conv(shift_tensor).view(T//tsm_length, tsm_length, split_size//2, H, W)
        pre_tensor = self.pre_conv(shift_tensor).view(T//tsm_length, tsm_length, split_size//4, H, W)
        post_tensor = self.post_conv(shift_tensor).view(T//tsm_length, tsm_length, split_size//4, H, W)
        main_tensor = main_tensor.view(T//tsm_length, tsm_length, C - split_size, H, W)

        if self.version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif self.version == 'circulant':
            pre_conv_tensor  = torch.cat((pre_conv_tensor [:, -1:  , ...],  # NOQA
                                     pre_conv_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_conv_tensor = torch.cat((post_conv_tensor[:,  1:  , ...],  # NOQA
                                     post_conv_tensor[:,   :1 , ...]), dim=1)  # NOQA
        return torch.cat((pre_tensor, post_tensor, main_conv_tensor, main_tensor), dim=2).view(shape)

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, tensor):
        # not support higher order gradient
        # tensor = tensor.detach_()
        t, c, h, w = tensor.size()
        n = 1
        tensor = tensor.view(1, t,c,h,w)
        fold = c // 4
        ctx.fold_ = fold
        buffer_ = tensor.data.new(n, t, fold, h, w).zero_()
        buffer_[:, :-1] = tensor.data[:, 1:, :fold]
        tensor.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, 1:] = tensor.data[:, :-1, fold: 2 * fold]
        tensor.data[:, :, fold: 2 * fold] = buffer_
        return tensor.view(t,c,h,w)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        t, c, h, w = grad_output.size()
        n = 1
        grad_output = grad_output.view(1, t,c,h,w)
        buffer_ = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer_[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer_
        return grad_output.view(t,c,h,w), None


def tsm_module(tensor, version='zero', inplace=True, tsm_length=3):
    if not inplace:
        shape = T, C, H, W = tensor.shape
        tensor = tensor.view(T//tsm_length, tsm_length, C, H, W)
        split_size = C // 4
        pre_tensor, post_tensor, peri_tensor = tensor.split(
            [split_size, split_size, C - 2 * split_size],
            dim=2
        )
        if version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif version == 'circulant':
            pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],  # NOQA
                                     pre_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_tensor = torch.cat((post_tensor[:,  1:  , ...],  # NOQA
                                     post_tensor[:,   :1 , ...]), dim=1)  # NOQA
        else:
            raise ValueError('Unknown TSM version: {}'.format(version))
        return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)
    else:
        out = InplaceShift.apply(tensor)
        return out
    

class conv_block(nn.Module):
    def __init__(self,in_chn,out_chn, tsm=False, alpha=1, learn=False, tsm_length=3, version='zero'):
        super(conv_block,self).__init__()
        
        self.tsm = tsm
        self.learn = learn
        self.tsm_length = tsm_length

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chn, out_channels=out_chn, kernel_size=3, stride=1, padding=1,bias=True),
            nn.ReLU()
            )
        
        if self.tsm:
            if self.learn:
                self.tsmConv = learnTSM(in_channels = out_chn, alpha=alpha, version=version)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chn, out_channels=out_chn, kernel_size=3,stride=2,padding=1,bias=True), 
            nn.ReLU()
        )

    def forward(self,x):
        
        x = self.conv1(x)
        
        if self.tsm:
            if self.learn:
                x = self.tsmConv(x, tsm_length=self.tsm_length)
            else:
                x = tsm_module(x, 'zero', tsm_length=self.tsm_length).contiguous()
                
        x = self.conv2(x)
        return x

class up_block(nn.Module):
    def __init__(self, in_chn, out_chn, alpha = 1, tsm = False, learn=False, tsm_length=3, version='zero'):
        super(up_block,self).__init__()
        
        self.tsm = tsm
        self.learn = learn
        self.tsm_length = tsm_length

        if(out_chn == 3):
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_chn, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1,bias=True),
                )
            
        elif(out_chn == 64):
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_chn, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1,bias=True),
                nn.ReLU()
                )
        
        if self.tsm:
            if self.learn:
                self.tsmConv = learnTSM(in_channels = out_chn, alpha=alpha, version=version)

        self.conv2 = nn.PixelShuffle(2)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)

        if self.tsm:
            if self.learn:
                x = self.tsmConv(x, tsm_length=self.tsm_length)
                
            else:
                x = tsm_module(x, 'zero', tsm_length=self.tsm_length).contiguous()
                
        return x
    
class autoencoder_v3(nn.Module):
    def __init__(self, in_chn = 3, out_chn = 3, alpha = 1, tsm = True, learn = True, tsm_length=3):
        super().__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.tsm_length = tsm_length

        # Encoder
        self.inc = conv_block(in_chn=self.in_chn, out_chn=64, alpha = alpha, tsm_length=self.tsm_length)
        self.conv_1 = conv_block(in_chn=64, out_chn=64, alpha = alpha, tsm = tsm, learn =learn, tsm_length=self.tsm_length)
        self.conv_2 = conv_block(in_chn=64, out_chn=64, alpha = alpha, tsm = tsm, learn =learn, tsm_length=self.tsm_length)
        self.conv_3 = conv_block(in_chn=64, out_chn=64, alpha = alpha, tsm = tsm, learn =learn, tsm_length=self.tsm_length)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Decoder
        self.up_0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()    
            )
        self.up_1 = up_block(in_chn=64, out_chn = 64, alpha=alpha, tsm = tsm, learn=learn, tsm_length=self.tsm_length)
        self.up_2 = up_block(in_chn=64, out_chn = 64, alpha=alpha, tsm = tsm, learn=learn, tsm_length=self.tsm_length)
        self.up_3 = up_block(in_chn=64, out_chn = 64, alpha=alpha, tsm = tsm, learn=learn, tsm_length=self.tsm_length)
        self.up_4 = up_block(in_chn=64, out_chn = self.out_chn, alpha=alpha, tsm_length=self.tsm_length)

    def forward(self, x):
        '''
        Model of CTSM for depth = 2.
        '''

        shape = B, T, self.in_chn, H, W = x.shape
        
        # print(shape)

        x = x.reshape(B*T, self.in_chn, H, W)

        # print("reshape", x.shape)

        # encoding path
        x1 = self.inc(x)
        
        x2 = self.conv_1(x1)
       
        x3 = self.conv_2(x2)
        
        x4 = self.conv_4(x3)

        x4 = torch.clamp(x4, 0, 1)

        # decoder path
        x5 = self.up_0(x4)
        
        x6 = self.up_2(x5)
        
        x7 = self.up_3(x6)
        
        x8= self.up_4(x7)
        
        x = torch.clamp(x8, 0, 1)
    

        x = x.view(B, T, self.out_chn, H, W)
        return x, x4
    
