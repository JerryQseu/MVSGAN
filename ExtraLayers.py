import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import numpy as np
import STN as STN

class affine_layer(nn.Module):
    def __init__(self):
        super(affine_layer, self).__init__()
        # self.in_feature = in_feature
        # self.out_feature = out_feature
        # self.weight = nn.Parameter(torch.Tensor(1,3,4))
        #
        #
        # stdv = 1. / math.sqrt(self.weight.size(0))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input,imput2):

        input = input#.cpu().numpy()
        weight = imput2.view(1,3,4)

        # weight = self.weight
        weight = weight#.cpu().detach().numpy()
        out = STN.test_3d(input,weight)
        # x = torch.Tensor(out)
        out = out.permute(0,4,3,1,2)




        return out

def transform3D(image, affine_matrix):
    # grab the shape of the image
    B, H, W, D, C = image.shape
    M = affine_matrix

    # mesh grid generation
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    z = np.linspace(0, 1, D)
    x_t, y_t, z_t = np.meshgrid(x, y, z)

    # augment the dimensions to create homogeneous coordinates
    # reshape to (xt, yt, zt, 1)
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])
    # repeat to number of batches
    sampling_grid = np.resize(sampling_grid, (B, 4, H * W * D))

    # transform the sampling grid, i.e. batch multiply
    batch_grids = np.matmul(M, sampling_grid)  # the batch grid has the shape (B, 3, H*W*D)

    # reshape to (B, H, W, D, 3)
    batch_grids = batch_grids.reshape(B, 3, H, W, D)
    batch_grids = np.moveaxis(batch_grids, 1, -1)

    # bilinear resampler
    x_s = batch_grids[:, :, :, :, 0:1].squeeze()
    y_s = batch_grids[:, :, :, :, 1:2].squeeze()
    z_s = batch_grids[:, :, :, :, 2:3].squeeze()

    # rescale x, y and z to [0, W/H/D]
    x = ((x_s) * W)
    y = ((y_s) * H)
    z = ((z_s) * D)

    # for each coordinate we need to grab the corner coordinates
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1
    z0 = np.floor(z).astype(np.int64)
    z1 = z0 + 1

    # clip to fit actual image size
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)
    z0 = np.clip(z0, 0, D - 1)
    z1 = np.clip(z1, 0, D - 1)

    # grab the pixel value for each corner coordinate
    Ia = image[np.arange(B)[:, None, None, None], y0, x0, z0]
    Ib = image[np.arange(B)[:, None, None, None], y1, x0, z0]
    Ic = image[np.arange(B)[:, None, None, None], y0, x1, z0]
    Id = image[np.arange(B)[:, None, None, None], y1, x1, z0]
    Ie = image[np.arange(B)[:, None, None, None], y0, x0, z1]
    If = image[np.arange(B)[:, None, None, None], y1, x0, z1]
    Ig = image[np.arange(B)[:, None, None, None], y0, x1, z1]
    Ih = image[np.arange(B)[:, None, None, None], y1, x1, z1]

    # calculated the weighted coefficients and actual pixel value
    wa = (x1 - x) * (y1 - y) * (z1 - z)
    wb = (x1 - x) * (y - y0) * (z1 - z)
    wc = (x - x0) * (y1 - y) * (z1 - z)
    wd = (x - x0) * (y - y0) * (z1 - z)
    we = (x1 - x) * (y1 - y) * (z - z0)
    wf = (x1 - x) * (y - y0) * (z - z0)
    wg = (x - x0) * (y1 - y) * (z - z0)
    wh = (x - x0) * (y - y0) * (z - z0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=4)
    wb = np.expand_dims(wb, axis=4)
    wc = np.expand_dims(wc, axis=4)
    wd = np.expand_dims(wd, axis=4)
    we = np.expand_dims(we, axis=4)
    wf = np.expand_dims(wf, axis=4)
    wg = np.expand_dims(wg, axis=4)
    wh = np.expand_dims(wh, axis=4)

    # compute output
    image_out = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
    #image_out = image_out.astype(np.int64)

    return image_out

class PixelShuffle3D(nn.Module):
    """
    三维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 1

        out_depth = in_depth * self.upscale_factor
        # out_height = in_height * self.upscale_factor
        # out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, 1, 1, in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, in_height, in_width)
