'''
VoxelMorph
Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
YU Meng
Xiamen University
'''



import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal
import utils
import math


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channel, channel // ratio, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(channel // ratio, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channel, _, _ = x.shape
        avg_pooled = self.avg_pool(x).view(batch, channel)
        max_pooled = self.max_pool(x).view(batch, channel)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pooled)))
        max_out = self.fc2(self.relu1(self.fc1(max_pooled)))
        scale = self.sigmoid(avg_out + max_out).view(batch, channel, 1, 1)
        return x * scale
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pooled, max_pooled], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        if self.grid.device != flow.device:
            self.grid = self.grid.to(flow.device)  # Move grid to the same device as flow
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            # 结果与原始向量场相加，完成向量场的积分操作
        return vec


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        # 根据提供的维度动态选择合适的卷积类

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)  # 实例归一化
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class CustomBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(CustomBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        # out += identity  # Residual connection
        out = out + identity  # Residual connection
        out = self.relu(out)

        out = self.conv5(out)
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """
    Encoder Module
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)
        self.fusion_conv0 = conv(c, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )
        self.fusion_conv1 = conv(2 * c, 2 * c, 3, 1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * c + 16, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )
        self.fusion_conv2 = conv(4 * c, 4 * c, 3, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * c + 16, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )
        self.fusion_conv3 = conv(8 * c, 8 * c, 3, 1)
        self.conv_img1 = CustomBlock(1, 16)
        self.conv_img2 = CustomBlock(1, 16)
        self.conv_img3 = CustomBlock(1, 16)

    def forward(self, x, x1, x2, x3):

        out0 = self.conv0(x)

        out1 = self.conv1(out0)
        out1 = self.fusion_conv1(out1)  # Fusion layer
        out1_img = self.conv_img1(x1)
        out1 = torch.cat([out1, out1_img], dim=1)

        out2 = self.conv2(out1)
        out2 = self.fusion_conv2(out2)  # Fusion layer
        out2_img = self.conv_img2(x2)
        out2 = torch.cat([out2, out2_img], dim=1)

        out3 = self.conv3(out2)
        out3 = self.fusion_conv3(out3)  # Fusion layer
        out3_img = self.conv_img1(x3)
        out3 = torch.cat([out3, out3_img], dim=1)

        return [out0, out1, out2, out3]



class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.LeakyReLU(0.1)
    )


class Conv2dReLU(nn.Sequential):  # 调用父类，桉顺序添加到’Sequential‘容器中
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,  # 布尔值，表示是否使用批归一化
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm2d(out_channels)  # 实例归一化
        else:
            nm = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, nm, relu)

class feature_fusion_module(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(feature_fusion_module, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x_thisBranch, x_otherBranch):

        theta_x = self.theta(x_thisBranch)
        phi_x = self.phi(x_otherBranch)
        similarity_map = torch.sum(theta_x * phi_x, dim=1, keepdim=True)
        attention_weights = torch.sigmoid(similarity_map)
        weighted_phi = attention_weights * phi_x
        fused_features = self.fusion_conv(torch.cat([theta_x, weighted_phi], dim=1))
        # fused_features = torch.cat([theta_x, weighted_phi], dim=1)
        return fused_features

class sequential_pyramid_net(nn.Module):
    def __init__(self, inshape, flow_multiplier=1., in_channel=1, channels=16):
        super(sequential_pyramid_net, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels  # c = 16

        self.encoder_moving = Encoder(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='bilinear',
                                           align_corners=True)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        self.warp_INVERSE = nn.ModuleList()
        self.diff_INVERSE = nn.ModuleList()
        for i in range(4):
            self.warp_INVERSE.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff_INVERSE.append(VecInt([s // 2 ** i for s in inshape]))

        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c+32, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )
        # warp scale 2
        self.defconv4 = nn.Conv2d(8 * c, 2, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3 * 8 * c+32, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.cconv_3 = CConv(3 * 4 * c+32)

        # warp scale 1
        self.defconv3 = nn.Conv2d(3 * 4 * c+32, 2, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = ConvInsBlock(3 * 4 * c+32, 4 * c)

        self.upconv2 = UpConvBlock(3 * 4 * c+32, 2 * c, 4, 2)
        self.cconv_2 = CConv(3 * 2 * c+32)

        # warp scale 0
        self.defconv2 = nn.Conv2d(3 * 2 * c+32, 2, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        self.dconv2 = ConvInsBlock(3 * 2 * c+32, 2 * c)

        self.upconv1 = UpConvBlock(3 * 2 * c+32, c, 4, 2)
        self.cconv_1 = CConv(3 * c)

        # decoder layers
        self.defconv1 = nn.Conv2d(3 * c, 2, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))
        self.resize_1 = utils.ResizeTransform(2, len(inshape))
        self.resize_2 = utils.ResizeTransform(4, len(inshape))
        self.resize_3 = utils.ResizeTransform(8, len(inshape))

        # 初始化注意力模块
        self.channel_attention_module_1 = ChannelAttention(channel=16)
        self.channel_attention_module_2 = ChannelAttention(channel=48)
        self.channel_attention_module_3 = ChannelAttention(channel=80)
        self.channel_attention_module_4 = ChannelAttention(channel=144)
        self.spatial_attention_module = SpatialAttention()
        self.path1_block3_NLCross = feature_fusion_module(144)
        self.path2_block3_NLCross = feature_fusion_module(144)

    def forward(self, moving, fixed):
        moving1, fixed1 = self.resize_1(moving), self.resize_1(fixed)
        moving2, fixed2 = self.resize_2(moving), self.resize_2(fixed)
        moving3, fixed3 = self.resize_3(moving), self.resize_3(fixed)

        M1, M2, M3, M4 = self.encoder_moving(moving, moving1, moving2, moving3)  # (128,128) (64,64) (32,32) (16,16)
        F1, F2, F3, F4 = self.encoder_fixed(fixed, fixed1, fixed2, fixed3)
        M1 = self.spatial_attention_module(M1)
        M1 = self.channel_attention_module_1(M1)
        # #
        F1 = self.spatial_attention_module(F1)
        F1 = self.channel_attention_module_1(F1)
        # #
        M2 = self.spatial_attention_module(M2)
        M2 = self.channel_attention_module_2(M2)
        # #
        F2 = self.spatial_attention_module(F2)
        F2 = self.channel_attention_module_2(F2)
        # #
        M3 = self.spatial_attention_module(M3)
        M3 = self.channel_attention_module_3(M3)
        # #
        F3 = self.spatial_attention_module(F3)
        F3 = self.channel_attention_module_3(F3)
        # #
        M4 = self.spatial_attention_module(M4)
        M4 = self.channel_attention_module_4(M4)
        # #
        F4 = self.spatial_attention_module(F4)
        F4 = self.channel_attention_module_4(F4)


        M4 = self.path1_block3_NLCross(M4, F4)
        F4 = self.path2_block3_NLCross(F4, M4)


        # first dec layer- after equalization
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)
        w = self.diff[3](v)
        flow_0 = flow  #1/8



        # second dec layer
        D3 = self.upconv3(C4)
        flow = self.upsample_trilin(2 * (self.warp[3](flow, w) + w))
        warped = self.warp[2](M3, flow)
        C3 = self.cconv_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)

        flow = self.warp[2](flow, w) + w  # 32,32
        warped = self.warp[2](M3, flow)
        D3 = self.dconv3(C3)
        C3 = self.cconv_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)

        flow_1 = flow  #1/4

        # third dec layer
        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2 * (self.warp[2](flow, w) + w))  # 64,64
        warped = self.warp[1](M2, flow)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)

        flow_2 = flow  #1/2

        D1 = self.upconv1(C2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.cconv_1(F1, warped, D1)
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w   #1
        return flow_0,  flow_1, flow_2, flow


class bidirectional_average_net(nn.Module):
    def __init__(self, inshape, flow_multiplier=1, channels=16):
        super(bidirectional_average_net, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape
        c = self.channels  # c = 16
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))
        self.layers3 = sequential_pyramid_net((128,128))

    def forward(self, moving, fixed):
        flow_0,  flow_1, flow_2,flow_3= self.layers3(moving,fixed)
        flow_0_INVERSE, flow_1_INVERSE, flow_2_INVERSE,flow_3_INVERSE = self.layers3(fixed,moving)
        vecs_3 = (flow_3 - flow_3_INVERSE) / 2
        flow_3 = VecInt((128, 128))(vecs_3)
        y_moved = self.warp[0](moving, flow_3)
        return y_moved, flow_0, flow_1,flow_2, flow_3


if __name__ == '__main__':
    size = (2, 1, 128, 128)
    model = bidirectional_average_net(size[2:])
    A = torch.ones(size)
    B = torch.ones(size)
    C = torch.ones(size)
    y_moved, flow_0, flow_1,flow_2,flow_3 = model(A, B)
    print(y_moved.shape, flow_0.shape,flow_1.shape,flow_2.shape,flow_3.shape)
