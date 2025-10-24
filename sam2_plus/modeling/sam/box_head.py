import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

# copy from https://github.com/MCG-NJU/MixFormer/blob/main/lib/models/mixformer_cvt/utils.py#L21
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


# copy from https://github.com/MCG-NJU/MixFormer/blob/main/lib/models/mixformer_cvt/head.py#L7
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


# copy from https://github.com/MCG-NJU/MixFormer/main/8b87088f45af02c614a277176c63181261e93de3/lib/models/mixformer_cvt/head.py#L23
class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


# copy from https://github.com/MCG-NJU/MixFormer/blob/main/lib/models/mixformer_cvt/head.py#L98
class Pyramid_Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Pyramid_Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_tl = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_tl = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_br = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_br = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_tl2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_tl3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_br2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_br3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class Pyramid_Corner_Predictor_multi_box(Pyramid_Corner_Predictor):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        """
        Process the feature map with 3 different resolutions [B, inplanes, H//16, W//16], [B, inplanes//4, H//8, W//8], [B, inplanes//8, H//4, W//4].
        Predict the corner coordinates [B, 4].

        @inplanes: the channel num of input feature
        @channel: the channel num of inner feature.
        @feat_sz: the size of score map. Actually, feat_sz * stride = H = W
        @stride: the stride of the feature map to the original image. Actually, feat_sz * stride = H = W

        In Sam2's feature map, the highest feature resolution is H//4 , so feat_sz should be set to H // 4, and stride should be set to 4.
        """
        super(Pyramid_Corner_Predictor_multi_box, self).__init__(inplanes, channel, feat_sz, stride, freeze_bn)
        # handle input feature map with different resolution
        del self.conv5_tl
        del self.adjust3_tl
        del self.adjust4_tl
        # self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        
        del self.conv5_br
        del self.adjust3_br
        del self.adjust4_br
        # self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )

    def forward(self, x, hyper_in, return_dist=False, softmax=True):
        """ 
        @x: [B, inplanes, H_s//16, W_s//16]
        @hyper_in: the hyper feature map, [B, num_tokens, C], where C = channel // 8

        Return:
            [B, num_tokens, 4]
        """
        # score_map_tl, score_map_br = self.get_score_map(x)    # [B, 1, H_s//4, W_s//4]
        score_map_tl, score_map_br = self.get_score_map(x, hyper_in) # [B, num_tokens, H_s//4, W_s//4], [B, num_tokens, H_s//4, W_s//4]
        B, N = hyper_in.shape[:2]
        score_map_tl, score_map_br = rearrange(score_map_tl, 'b n h w -> (b n) 1 h w'), rearrange(score_map_br, 'b n h w -> (b n) 1 h w')   # [B * num_tokens, 1, H_s//4, W_s//4]

        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            # return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
            coorx = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx, prob_vec_tl, prob_vec_br = rearrange(coorx, '(b n) C -> b n C', b=B, n=N, C=4), rearrange(prob_vec_tl, '(b n) C -> b n C', b=B, n=N), rearrange(prob_vec_br, '(b n) C -> b n C', b=B, n=N)
            return coorx, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            # return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx = rearrange(coorx, '(b n) C -> b n C', b=B, n=N, C=4)
            return coorx

    def get_score_map(self, x, hyper_in):
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_tl2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_tl3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)

        # score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)
        score_map_tl = x_tl4 + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)    # [B, C = channel // 8, H_s//4, W_s//4]

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_br2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_br3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)

        # score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        score_map_br = x_br4 + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)    # [B, C = channel // 8, H_s//4, W_s//4]

        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        score_map_tl = torch.einsum("bnc,bchw->bnhw", hyper_in, score_map_tl)     # [B, num_tokens, H_s//4, W_s//4]
        score_map_br = torch.einsum("bnc,bchw->bnhw", hyper_in, score_map_br)     # [B, num_tokens, H_s//4, W_s//4]

        return score_map_tl, score_map_br


class Pyramid_Corner_Predictor_multi_box_use_high_res_features(Pyramid_Corner_Predictor):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        """
        Process the feature map with 3 different resolutions [B, inplanes, H//16, W//16], [B, inplanes//4, H//8, W//8], [B, inplanes//8, H//4, W//4].
        Predict the corner coordinates [B, 4].

        @inplanes: the channel num of input feature
        @channel: the channel num of inner feature.
        @feat_sz: the size of score map. Actually, feat_sz * stride = H = W
        @stride: the stride of the feature map to the original image. Actually, feat_sz * stride = H = W

        In Sam2's feature map, the highest feature resolution is H//4 , so feat_sz should be set to H // 4, and stride should be set to 4.
        """
        super(Pyramid_Corner_Predictor_multi_box_use_high_res_features, self).__init__(inplanes, channel, feat_sz, stride, freeze_bn)
        # handle input feature map with different resolution
        del self.conv5_tl
        del self.adjust3_tl
        del self.adjust4_tl
        # self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust_16_to_16_tl = conv(inplanes, inplanes, freeze_bn=freeze_bn)
        self.adjust_8_to_8_tl = conv(inplanes // 4, channel // 2, freeze_bn=freeze_bn)
        self.adjust_4_to_4_tl = conv(inplanes // 8, channel // 4, freeze_bn=freeze_bn)
        
        del self.conv5_br
        del self.adjust3_br
        del self.adjust4_br
        # self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        # conv(channel // 8, 1, freeze_bn=freeze_bn)
                                        )
        self.adjust_16_to_16_br = conv(inplanes, inplanes, freeze_bn=freeze_bn)
        self.adjust_8_to_8_br = conv(inplanes // 4, channel // 2, freeze_bn=freeze_bn)
        self.adjust_4_to_4_br = conv(inplanes // 8, channel // 4, freeze_bn=freeze_bn)

    def forward(self, x_list, hyper_in, return_dist=False, softmax=True):
        """ 
        @x_list: list of feature maps, [B, inplanes, H_s//16, W_s//16], [B, inplanes//4, H_s//8, W_s//8], [B, inplanes//8, H_s//4, W_s//4]
        @hyper_in: the hyper feature map, [B, num_tokens, C], where C = channel // 8

        Return:
            [B, num_tokens, 4]
        """
        # score_map_tl, score_map_br = self.get_score_map(x)    # [B, 1, H_s//4, W_s//4]
        score_map_tl, score_map_br = self.get_score_map(x_list, hyper_in) # [B, num_tokens, H_s//4, W_s//4], [B, num_tokens, H_s//4, W_s//4]
        B, N = hyper_in.shape[:2]
        score_map_tl, score_map_br = rearrange(score_map_tl, 'b n h w -> (b n) 1 h w'), rearrange(score_map_br, 'b n h w -> (b n) 1 h w')   # [B * num_tokens, 1, H_s//4, W_s//4]

        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            # return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
            coorx = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx, prob_vec_tl, prob_vec_br = rearrange(coorx, '(b n) C -> b n C', b=B, n=N, C=4), rearrange(prob_vec_tl, '(b n) C -> b n C', b=B, n=N), rearrange(prob_vec_br, '(b n) C -> b n C', b=B, n=N)
            return coorx, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            # return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz
            coorx = rearrange(coorx, '(b n) C -> b n C', b=B, n=N, C=4)
            return coorx
   
    def get_score_map(self, x_list: list, hyper_in: torch.Tensor):
        """
        Input:
            @x_list:  list of feature maps, [B, inplanes, H_s//16, W_s//16], [B, inplanes//4, H_s//8, W_s//8], [B, inplanes//8, H_s//4, W_s//4]
            @hyper_in: the hyper feature map, [B, num_tokens, c], where c = channel // 8
        Return:
            score_map: [B, num_tokens, H_s//4, W_s//4]
        """
        x_init_4, x_init_8, x_init_16 = x_list

        assert x_init_4.shape[-1] == self.feat_sz, "the highest resolution feature map should have the same size as feat_sz"

        ###### top-left branch ######
        x_init_tl = self.adjust_16_to_16_tl(x_init_16)                          # [B, inplanes, H_s//16, W_s//16] -> [B, inplanes, H_s//16, W_s//16]

        x_tl1 = self.conv1_tl(x_init_tl)                                        # [B, inplanes, H_s//16, W_s//16] -> [B, channel, H_s//16, W_s//16]
        x_tl2 = self.conv2_tl(x_tl1)                                            # [B, channel, H_s//16, W_s//16] -> [B, channel//2, H_s//16, W_s//16]

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init_tl), scale_factor=2)  # [B, inplanes, H_s//16, W_s//16] -> [B, channel//2, H_s//8, W_s//8]
        x_up1 = F.interpolate(x_tl2, scale_factor=2)                            # [B, channel//2, H_s//16, W_s//16] -> [B, channel//2, H_s//8, W_s//8]
        x_init_8_adjust = self.adjust_8_to_8_tl(x_init_8)                       # [B, inplanes//4, H_s//8, W_s//8] -> [B, channel//2, H_s//8, W_s//8]
        x_up1 = x_init_up1 + x_up1 + x_init_8_adjust                            # [B, channel//2, H_s//8, W_s//8]

        x_tl3 = self.conv3_tl(x_up1)                                            # [B, channel//2, H_s//8, W_s//8] -> [B, channel//4, H_s//8, W_s//8]

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init_tl), scale_factor=4)  # [B, inplanes, H_s//16, W_s//16] -> [B, channel//4, H_s//4, W_s//4]
        x_up2 = F.interpolate(x_tl3, scale_factor=2)                            # [B, channel//4, H_s//8, W_s//8] -> [B, channel//4, H_s//4, W_s//4]
        x_init_4_adjust = self.adjust_4_to_4_tl(x_init_4)                       # [B, inplanes//8, H_s//4, W_s//4] -> [B, channel//4, H_s//4, W_s//4]
        x_up2 = x_init_up2 + x_up2 + x_init_4_adjust                            # [B, channel//4, H_s//4, W_s//4]

        x_tl4 = self.conv4_tl(x_up2)                                            # [B, channel//4, H_s//4, W_s//4] -> [B, channel//8, H_s//4, W_s//4]

        # score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)    # [B, 1, H_s//4, W_s//4]
        score_map_tl = x_tl4 + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)    # [B, channel//8, H_s//4, W_s//4]

        ###### bottom-right branch ######
        x_init_br = self.adjust_16_to_16_br(x_init_16)                          # [B, inplanes, H_s//16, W_s//16] -> [B, inplanes, H_s//16, W_s//16]

        x_br1 = self.conv1_br(x_init_br)                                        # [B, inplanes, H_s//16, W_s//16] -> [B, channel, H_s//16, W_s//16]
        x_br2 = self.conv2_br(x_br1)                                            # [B, channel, H_s//16, W_s//16] -> [B, channel//2, H_s//16, W_s//16]

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init_br), scale_factor=2)  # [B, inplanes, H_s//16, W_s//16] -> [B, channel//2, H_s//8, W_s//8]
        x_up1 = F.interpolate(x_br2, scale_factor=2)                            # [B, channel//2, H_s//16, W_s//16] -> [B, channel//2, H_s//8, W_s//8]
        x_init_8_adjust = self.adjust_8_to_8_br(x_init_8)                       # [B, inplanes//4, H_s//8, W_s//8] -> [B, channel//2, H_s//8, W_s//8]
        x_up1 = x_init_up1 + x_up1 + x_init_8_adjust                            # [B, channel//2, H_s//8, W_s//8]

        x_br3 = self.conv3_br(x_up1)                                            # [B, channel//2, H_s//8, W_s//8] -> [B, channel//4, H_s//8, W_s//8]

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init_br), scale_factor=4)  # [B, inplanes, H_s//16, W_s//16] -> [B, channel//4, H_s//4, W_s//4]
        x_up2 = F.interpolate(x_br3, scale_factor=2)                            # [B, channel//4, H_s//8, W_s//8] -> [B, channel//4, H_s//4, W_s//4]
        x_init_4_adjust = self.adjust_4_to_4_br(x_init_4)                       # [B, inplanes//8, H_s//4, W_s//4] -> [B, channel//4, H_s//4, W_s//4]
        x_up2 = x_init_up2 + x_up2 + x_init_4_adjust                            # [B, channel//4, H_s//4, W_s//4]

        x_br4 = self.conv4_br(x_up2)                                            # [B, channel//4, H_s//4, W_s//4] -> [B, channel//8, H_s//4, W_s//4]

        # score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        score_map_br = x_br4 + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)    # [B, channel//8, H_s//4, W_s//4]

        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        score_map_tl = torch.einsum("bnc,bchw->bnhw", hyper_in, score_map_tl)     # [B, num_tokens, H_s//4, W_s//4]
        score_map_br = torch.einsum("bnc,bchw->bnhw", hyper_in, score_map_br)     # [B, num_tokens, H_s//4, W_s//4]

        return score_map_tl, score_map_br