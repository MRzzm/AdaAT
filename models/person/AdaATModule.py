import torch
from torch import nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

def make_coordinate_grid(spatial_size, type):
    '''
    generate 2D coordinate grid
    '''
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

def make_coordinate_grid_3d(spatial_size, type):
    '''
    generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

def kp2gaussian(landmark, spatial_size, kp_variance):
    '''
    generate heatmap from 2d key points
    '''
    coordinate_grid = make_coordinate_grid(spatial_size, landmark.type())
    number_of_leading_dimensions = len(landmark.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = landmark.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    shape = landmark.shape[:number_of_leading_dimensions] + (1, 1, 2)
    landmark = landmark.view(*shape)
    mean_sub = (coordinate_grid - landmark)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out

class ResBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    '''
    basic block
    '''
    def __init__(self, num_channels, num_down_blocks=3, block_expansion=64, max_features=512,
                 ):
        super(Encoder, self).__init__()
        self.in_conv = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.Sequential(*down_blocks)
    def forward(self, image):
        out = self.in_conv(image)
        out = self.down_blocks(out)
        return out

class Decoder(nn.Module):
    '''
    basic block
    '''
    def __init__(self,num_channels, num_down_blocks=3, block_expansion=64, max_features=512):
        super(Decoder, self).__init__()
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_conv = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.sigmoid = nn.Sigmoid()
    def forward(self, feature_map):
        out = self.up_blocks(feature_map)
        out = self.out_conv(out)
        out = self.sigmoid(out)
        return out

class AdaAT(nn.Module):
    '''
    Our proposed AdaAT operation
    '''
    def __init__(self,  para_ch,feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
                    nn.Linear(para_ch, feature_ch),
                    nn.Sigmoid()
                )
        self.rotation = nn.Sequential(
                nn.Linear(para_ch, feature_ch),
                nn.Tanh()
            )
        self.translation = nn.Sequential(
                nn.Linear(para_ch, 2 * feature_ch),
                nn.Tanh()
            )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map,para_code):
        batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        ######## compute affine trans parameters
        para_code = self.commn_linear(para_code)
        # compute scale para
        scale = self.scale(para_code).unsqueeze(-1) * 2
        # compute rotation angle
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159
        # transform rotation angle to ratation matrix
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        # compute translation para
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        ########  do affine transformation
        # compute 3d coordinate grid
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        # do affine transformation on channels
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        # interpolation
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature

class AdaIN(nn.Module):
    def __init__(self,  para_ch,feature_ch,  eps=1e-5):
        super(AdaIN, self).__init__()
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.mean = nn.Linear(para_ch, feature_ch)
        self.var = nn.Linear(para_ch, feature_ch)
        self.eps = eps

    def forward(self, feature_map,para_code):
        batch, d = feature_map.size(0), feature_map.size(1)
        para_code = self.commn_linear(para_code)
        ########### compute mean and var
        mean = self.mean(para_code).unsqueeze(2).unsqueeze(3)
        var = self.var(para_code).unsqueeze(2).unsqueeze(3)
        ########## normalization
        feature_var = torch.var(feature_map.view(batch, d, -1), dim=2, keepdim=True).unsqueeze(-1) + self.eps
        feature_std = feature_var.sqrt()
        feature_mean = torch.mean(feature_map.view(batch, d, -1), dim=2, keepdim=True).unsqueeze(-1)
        feature_map = (feature_map - feature_mean) / feature_std
        norm_feature = feature_map * var + mean
        return norm_feature


class AdaATModule(nn.Module):
    def __init__(self, img_channel,keypoint_num,  kp_variance = 0.01):
        super(AdaATModule, self).__init__()
        self.appearance_encoder = nn.Sequential(
            Encoder(img_channel + keypoint_num * 2, num_down_blocks=2, block_expansion=64, max_features=256),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256,512, 3, 1),
            ResBlock2d(512,512, 3, 1),
        )
        self.trans_encoder = nn.Sequential(
            DownBlock2d(512,256,3),
            DownBlock2d(256, 256, 3),
            DownBlock2d(256, 256, 3),
            DownBlock2d(256, 256, 3)
        )
        appearance_conv_list = []
        for i in range(3):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(512, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 512, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)

        self.adaAT1 = AdaAT(256,512)
        self.adaAT2 = AdaAT(256, 512)
        self.adaAT3 = AdaIN(256, 512)

        self.appearance_decoder = nn.Sequential(
            ResBlock2d(512, 512, 3, 1),
            ResBlock2d(512, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            Decoder(img_channel, num_down_blocks=2, block_expansion=64, max_features=512)
        )
        self.kp_variance = kp_variance
        self.global_avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, source_image,source_kp,target_kp):
        batch,h,w = source_image.size(0),source_image.size(2),source_image.size(3)
        # transform 2d key points to heatmap
        source_heatmap = kp2gaussian(source_kp,(h,w),self.kp_variance)
        target_heatmap = kp2gaussian(target_kp,(h,w),self.kp_variance)
        # concat input data
        module_in = torch.cat([source_image,source_heatmap,target_heatmap],1)
        # compute appearance feature map
        appearance_feature = self.appearance_encoder(module_in)
        ######################## transformation branch
        # compute paras of affine transformation
        para_code = self.trans_encoder(appearance_feature)
        para_code = self.global_avg(para_code).squeeze(3).squeeze(2)
        ######################## feature alignment
        appearance_feature = self.appearance_conv_list[0](appearance_feature)
        appearance_feature = self.adaAT1(appearance_feature,para_code)
        appearance_feature = self.appearance_conv_list[1](appearance_feature)
        appearance_feature = self.adaAT2(appearance_feature,para_code)
        appearance_feature = self.appearance_conv_list[2](appearance_feature)
        appearance_feature = self.adaAT3(appearance_feature,para_code)
        # decode output image
        out = self.appearance_decoder(appearance_feature)
        return out,target_heatmap

    def visualize_feature(self,source_image,source_kp,target_kp):
        # visualize feature maps before and after 1𝑠𝑡  AdaAT
        batch, h, w = source_image.size(0), source_image.size(2), source_image.size(3)
        source_heatmap = kp2gaussian(source_kp, (h, w), self.kp_variance)
        target_heatmap = kp2gaussian(target_kp, (h, w), self.kp_variance)
        module_in = torch.cat([source_image, source_heatmap, target_heatmap], 1)
        appearance_feature = self.appearance_encoder(module_in)
        ## transformation branch
        para_code = self.trans_encoder(appearance_feature)
        para_code = self.global_avg(para_code).squeeze(3).squeeze(2)
        ## feature alignment
        trans_feature1 = self.appearance_conv_list[0](appearance_feature)
        trans_feature1_afterwarp = self.adaAT1(trans_feature1, para_code)
        return trans_feature1, trans_feature1_afterwarp
