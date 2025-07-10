# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from odl.contrib import torch as odl_torch
from basicsr.utils import compute_sinogram
fp, fbp = compute_sinogram(img_size=256, angle=360)
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

# Simplified Channel Attention
class SCABlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super(SCABlock, self).__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias)
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
    def forward(self,x):
        res = self.body(x)
        res = res*self.sca(res)
        res += x
        return res

#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation=nn.ReLU()):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = activation
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)

        # Concatenate along channel axis
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, 2, height, width)
        
        # Apply 1x1 convolution to generate spatial attention map
        attention_map = self.conv(spatial_features)  # (batch_size, 1, height, width)
        attention_map = self.sigmoid(attention_map)  # Apply sigmoid to scale between 0 and 1

        # Apply attention map to input
        return x * attention_map  # Element-wise multiplication

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# MDPRBlock
class MDPRBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Channel Attention
        self.ca = CAB(dw_channel // 2, dw_channel // 2)

        # Spatial attention
        self.sa = SpatialAttention()

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.ca(x)
        x = self.sa(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation=nn.ReLU()):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = activation
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi


class SSFNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, bias, num_cab):
        super(SSFNet, self).__init__()

        self.orb1 = nn.Sequential(*[MDPRBlock(n_feat+scale_orsnetfeats) for _ in range(num_cab)])
        self.orb2 = nn.Sequential(*[MDPRBlock(n_feat+scale_orsnetfeats) for _ in range(num_cab)])
        self.orb3 = nn.Sequential(*[MDPRBlock(n_feat+scale_orsnetfeats) for _ in range(num_cab)])
        

        self.up_enc1 = UpSample(n_feat*2)
        self.up_dec1 = UpSample(n_feat*2)

        self.up_enc2 = nn.Sequential(UpSample(n_feat*4), UpSample(n_feat*2))
        self.up_dec2 = nn.Sequential(UpSample(n_feat*4), UpSample(n_feat*2))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class Stage1UNet(nn.Module):

    def __init__(self, img_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[MDPRBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.lgag = LGAG(chan, chan, chan // 2)

        self.middle_blks = \
            nn.Sequential(
                *[MDPRBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[MDPRBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        # x = inp

        encs = []
        decs = []

        # Process features of both 2 patches with Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x_v = torch.flip(x, dims=[3])
        x = self.lgag(x_v, x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            decs.append(x)
        
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W], encs, decs[::-1]
        # return encs, decs[::-1]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Stage0UNet(nn.Module):

    def __init__(self, width, middle_blk_num, enc_blk_nums, dec_blk_nums):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[MDPRBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[MDPRBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[MDPRBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = inp
        encs = []

        # Process features of both 2 patches with Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class MDPRNet(nn.Module):
    def __init__(self, middle_blk_num1, enc_blk_nums1, dec_blk_nums1, middle_blk_num0, enc_blk_nums0, dec_blk_nums0, num_cab,
                in_c=3, out_c=3, stage0_width=32, stage1_width=64, stage2_width=32, kernel_size=3, bias=False):
        super(MDPRNet, self).__init__()

        act=nn.ReLU()
        self.shallow_feat0 = nn.Sequential(conv(in_c, stage0_width, kernel_size, bias=bias), SCABlock(stage0_width,kernel_size, bias=bias, act=act))
        self.shallow_feat1 = nn.Sequential(conv(in_c, stage1_width, kernel_size, bias=bias), SCABlock(stage1_width,kernel_size, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, stage2_width, kernel_size, bias=bias), SCABlock(stage2_width,kernel_size, bias=bias, act=act))
        
        self.fp = odl_torch.OperatorModule(fp)
        self.fbp = odl_torch.OperatorModule(fbp)

        # Cross Stage Feature Fusion (CSFF)
        self.stage0 = Stage0UNet(width=stage0_width, middle_blk_num=middle_blk_num0, enc_blk_nums=enc_blk_nums0, 
                                dec_blk_nums=dec_blk_nums0)
        self.stage1 = Stage1UNet(img_channel=in_c, width=stage1_width, middle_blk_num=middle_blk_num1, enc_blk_nums=enc_blk_nums1, 
                                dec_blk_nums=dec_blk_nums1)
        self.orsnet = SSFNet(stage1_width, stage2_width, bias, num_cab)


        self.sam01 = SAM(stage0_width, kernel_size=1, bias=bias)
        self.sam12 = SAM(stage1_width, kernel_size=1, bias=bias)
        
        self.concat01  = conv(stage0_width+stage1_width, stage1_width, kernel_size, bias=bias)
        self.concat12  = conv(stage2_width+stage1_width, stage2_width+stage1_width, kernel_size, bias=bias)
        self.tail     = conv(stage2_width+stage1_width, out_c, kernel_size, bias=bias)

    def forward(self, x2_img):

        ## Stage 0
        img_sino = self.fp(x2_img)   # torch.Size([2, 3, 360, 768])
        img_sino = self.shallow_feat0(img_sino)
        stage0_out = self.stage0(img_sino)
        stage0_out = self.fbp(stage0_out)
        stage0_feat = self.shallow_feat0(x2_img)
        stage0_out = self.stage0(stage0_feat)


        ## Apply Supervised Attention Module (SAM)
        stage0_samfeats, stage0_img = self.sam01(stage0_out, x2_img)

        ## Stage 1
        # stage1_feat = self.shallow_feat1(x2_img)
        # stage1_cat = self.concat01(torch.cat([stage1_feat, stage0_samfeats],dim=1))
        stage1_img, encs_feature, decs_feature = self.stage1(x2_img) 
        ## Apply Supervised Attention Module (SAM)
        stage1_samfeats, stage1_img = self.sam12(decs_feature[0], x2_img)
        
        ## Stage 2
        stage2_feat = self.shallow_feat2(x2_img) # x2[b, n_feat, 256, 256]
        stage2_cat = self.concat12(torch.cat([stage2_feat, stage1_samfeats], 1))
        x2_cat = self.orsnet(stage2_cat, encs_feature, decs_feature)
        stage2_img = self.tail(x2_cat)

        return stage2_img+x2_img, stage1_img, stage0_img


if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = MDPRNet(middle_blk_num1=middle_blk_num, enc_blk_nums1=enc_blks, dec_blk_nums1=dec_blks, middle_blk_num0=middle_blk_num, enc_blk_nums0=enc_blks, dec_blk_nums0=dec_blks, num_cab=8)
    inp = torch.randn(2,3,256,256)
    out = net(inp)
    print(len(out))


    # inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)