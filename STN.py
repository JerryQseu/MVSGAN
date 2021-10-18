from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0")

def test_3d(input_fmap, theta, out_dims=None, **kwargs):
    # input: (B, C, D, H, W)
    # theta: (B, 12)
    # grab input dimensions
    B = input_fmap.size()[0]
    H = input_fmap.size()[3]
    W = input_fmap.size()[4]
    D = input_fmap.size()[2]
    #theta[:,1,2]=1
    #theta[:,1,3]=1
    #theta[:,2,0]=1

    # reshape theta to (B, 3, 4), 刚好B=1, 如果B不为1的话，那么得修改下
    theta = torch.reshape(theta, (B, 3, 4))

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        out_D = out_dims[2]
        batch_grids = affine_grid_generator_3d(out_H, out_W, out_D, theta)
    else:
        # source grid (num_batch, 3, H, W, D) 到grid应该都没问题。
        batch_grids = affine_grid_generator_3d(H, W, D, theta)

    # source grid
    # print("the size of batch_grid before transpose is {} ".format(batch_grids.size()))
    batch_grids = batch_grids.permute(0, 1, 3, 2, 4)
    x_s = batch_grids[:, 0, :, :, :]  # （1， 320， 240， 10）
    y_s = batch_grids[:, 1, :, :, :]
    z_s = batch_grids[:, 2, :, :, :]



    # (B, H, W, D, C)
    input_fmap = input_fmap.permute(0, 3, 4, 2, 1)
    out = bilinear_sampler_3d_jchen(input_fmap, x_s, y_s, z_s)
    return out



def affine_grid_generator_3d(height, width, depth, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - depth: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 3, 4).
      For each image in the batch, we have 12 theta parameters of
      the form (3x4) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 3, H, W, D).
      The 2nd dimension has 3 components: (x, y, z) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    # theta (1, 3, 4)
    num_batch = theta.size()[0]

    # create normalized 2D grid, target grid
    x = torch.linspace(-1.0, 1.0, width)
    y = torch.linspace(-1.0, 1.0, height)
    z = torch.linspace(-1.0, 1.0, depth)
    # the shape of each x_t, y_t, z_t (W, H, D), verified
    x_t, y_t, z_t = torch.meshgrid(x, y, z)

    # flatten to 1-D (W*H*D, )
    x_t_flat = torch.reshape(x_t, (-1,))
    y_t_flat = torch.reshape(y_t, (-1,))
    z_t_flat = torch.reshape(z_t, (-1,))

    ones = torch.ones_like(x_t_flat)

    # stack to [x_t, y_t, z_t , 1] - (homogeneous form of target grid)
    sampling_grid = torch.stack((x_t_flat, y_t_flat, z_t_flat, ones))

    # repeat grid num_batch times, 增加维度
    sampling_grid = torch.unsqueeze(sampling_grid, 0)  # (1, 4, h*w*d)

    # 第0维度复制num_batch次，第1维度复制1次，第2维度复制1次
    # stack_a = torch.stack((num_batch, 1, 1))
    # stack_a = torch.IntTensor([num_batch, 1, 1]).numpy()
    sampling_grid = sampling_grid.repeat((num_batch, 1, 1))  # (1, 4, h*w*d), 在这里复制1次

    # cast to float32 (required for matmul)
    theta = theta.float()
    sampling_grid = sampling_grid.float().to(device)
    # print("------------")
    # # (1, 3, 4) matmul (1, 4, h*w*d)
    # print(sampling_grid.size())
    # print(theta.size())
    # transform the sampling grid - batch multiply;
    # batch_grid has shape (num_batch, 3, W*H*D)
    batch_grids = theta.matmul(sampling_grid)

    # reshape to (num_batch, 3, W, H, D), 之后还有 permute到 （N, C, D, H, W)
    batch_grids = torch.reshape(batch_grids, (num_batch, 3, width, height, depth))

    return batch_grids

def bilinear_sampler_3d_jchen(img, x, y, z):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, D, H, W, C) layout.
    - grid: x, y, z which is the output of affine_grid_generator.
            x, y, z should have the most values of [-1, 1]
            x_s, y_s, z_s : (B, H, W, D)
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
            Tensor of size (B, H, W, D, C)
    """
    # print("input for sampling is {}".format(img.size())) #(B, C, D, H, W)
    # print("input grid is {}".format(x.size()))
    H = img.size()[1]
    W = img.size()[2]
    D = img.size()[3]
    max_y = H-1
    max_x = W-1
    max_z = D-1
    # zero = torch.zeros([], dtype=torch.int32)
    zero = 0

    # rescale the value of x, y and z to [0, W-1/H-1]
    x = x.float()
    y = y.float()
    z = z.float()
    x = 0.5 * ((x + 1.0) * float(max_x))
    y = 0.5 * ((y + 1.0) * float(max_y))
    z = 0.5 * ((z + 1.0) * float(max_z))

    # grab 4 nearest corner points for each (x_i, y_i) and 2 nearest z-plane, make of 8 nearest corner points
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)
    z0 = torch.clamp(z0, zero, max_z)
    z1 = torch.clamp(z1, zero, max_z)

    # get pixel value at corner coords Ia, ...
    # (B, H, W, D, 3), 这边的值要为整数，因为代表了下标。
    I000 = get_pixel_value_3d(img, x0, y0, z0)
    I001 = get_pixel_value_3d(img, x0, y0, z1)
    I010 = get_pixel_value_3d(img, x0, y1, z0)
    I011 = get_pixel_value_3d(img, x0, y1, z1)
    I100 = get_pixel_value_3d(img, x1, y0, z0)
    I101 = get_pixel_value_3d(img, x1, y0, z1)
    I110 = get_pixel_value_3d(img, x1, y1, z0)
    I111 = get_pixel_value_3d(img, x1, y1, z1)

    dx = x - x0.float()
    dy = y - y0.float()
    dz = z - z0.float()
    # (B, H, W, D, 1)
    w000 = torch.unsqueeze((1. - dx) * (1. - dy) * (1. - dz), dim=4)
    w001 = torch.unsqueeze((1. - dx) * (1. - dy) * dz, dim=4)
    w010 = torch.unsqueeze((1. - dx) * dy * (1. - dz), dim=4)
    w011 = torch.unsqueeze((1. - dx) * dy * dz, dim=4)
    w100 = torch.unsqueeze(dx * (1. - dy) * (1. - dz), dim=4)
    w101 = torch.unsqueeze(dx * (1. - dy) * dz, dim=4)
    w110 = torch.unsqueeze(dx * dy * (1. - dz), dim=4)
    w111 = torch.unsqueeze(dx * dy * dz, dim=4)

    # compute output, 一个列表元素相加, 距离加权对应位置像素，得到最后结果，(B, H, W, D, C)
    out = w000 * I000 + w001 * I001 + w010 * I010 + w011 * I011 + w100 * I100 + w101 * I101 + w110 * I110 + w111 * I111

    return out

# I am not sure if the F.grid_sample works
def get_pixel_value_3d(img, x, y, z):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, D, C)
    Comment (JChen):
    源来是写flatten
    I think the shape of x, y is non-flattened, and thus (B, H, W, D)
    即： 每个x上点的值都代表在目标像素x coor上的索引值！
    Returns
    -------
    - output: tensor of shape (B, H, W, D, C)
    """
    shape = x.size()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]
    width_tensor = torch.FloatTensor([(width-1)*0.5]).to(device)
    height_tensor = torch.FloatTensor([(height-1)*0.5]).to(device)
    depth_tensor = torch.FloatTensor([(depth-1)*0.5]).to(device)
    one = torch.FloatTensor([1.0]).to(device)

    x = x.float() / width_tensor - one
    y = y.float() / height_tensor -one
    z = z.float() / depth_tensor - one

    indices_test = torch.stack((x, y, z), 4)
    indices_test = indices_test.permute(0, 3, 1, 2, 4).float()  # grid_sample 的输入必须是 float
    img = img.permute(0, 4, 3, 1, 2)
    # 多维索引， 取值； indices 上 每个
    # gather_nd(B, H, W, D, C), 多维索引！  indices (B, H, W, D, 4), 甚至要对B索引,如果不对B索引
    # gather = tf.gather_nd(img, indices)  # ---------------test
    # (N, C, D, H, W)
    gather_test = F.grid_sample(img, indices_test)  # ------------ test
    # return (N, C, D, H, W), while expected: (B, H, W, D, C)
    gather = gather_test.float()
    gather = gather.permute(0, 3, 4, 2, 1)
    return gather


# def read_image():
#     f = glob.iglob(r'*.jpg')
#     img = []
#     # f is a generator
#     for i in f:
#         img.append(np.array(Image.open(i)))
#         # (5, 28, 28, 1)
#     #if np.shape(img[3])==None:
#     # img = tf.expand_dims(img, -1)
#     print(np.shape(img))
#     return img

def read_theta(theta, W, H, D):
    ## return 6 points
    x_min = round((W / 2) * (1 - theta[0] + theta[3]))
    x_max = round((W / 2) * (1 + theta[0] + theta[3]))
    y_min = round((H / 2) * (1 - theta[1] + theta[4]))
    y_max = round((H / 2) * (1 + theta[1] + theta[4]))
    z_min = round((D / 2) * (1 - theta[2] + theta[5]))
    z_max = round((D / 2) * (1 + theta[2] + theta[5]))
    return x_min, x_max, y_min, y_max, z_min, z_max

# if __name__ == '__main__':
#
#     from PIL import Image, ImageDraw
#     import cv2
#     import glob
#
#
#     ######### 3D 验证 -- RGB
#     imgg = read_image()
#     # (5, 240, 320, 3)
#     d, h, w, c = np.shape(imgg)
#     # input: (C, D, H, W)
#     imgg = np.transpose(imgg, [3, 0, 1, 2])
#
#     # plot = imgg[:, :, 8, :].astype("uint8")
#     # plot = Image.fromarray(plot, 'RGB')
#     # plot.show()
#
#     # input: (B, C, D, H, W)
#     x_np = np.expand_dims(imgg, axis=0)
#     print("shape")
#     print(np.shape(x_np))
#     x_tensor = torch.tensor(x_np).float()
#     print("the size of x_tensor (input) is {} ".format(x_tensor.size()))
#
#
#     # test for original img
#     theta = [0.5, 0.5, 0.5, -0.3, 0.2, 0]
#     x_min, x_max, y_min, y_max, z_min, z_max = read_theta(theta, w, h, d)
#     x_plot = x_np[0, :, 5, :, :] # (1, 3, 9, 240, 320) --> (3, 240, 320)
#     x_plot = np.transpose(x_plot, (1, 2, 0))
#     print(np.shape(x_plot))
#     im = Image.fromarray(x_plot, 'RGB')
#     draw = ImageDraw.Draw(im)
#     draw.rectangle((x_min, y_min, x_max, y_max), outline='red')
#     im.show()
#     # (1, 12)
#
#     identity_matrix = torch.FloatTensor([[0.5, 0, 0, -0.3, 0, 0.5, 0, 0.2, 0, 0, 1, 0]])
#     out = test_3d(x_tensor, identity_matrix)  # in fact (B, H, W, D, C)
#     print("----------")
#     print(out.size())
#
#     img = out[0, :, :, 5, :].numpy()  # (H, W, C)
#     # # print(np.shape(img))
#     img0 = img.astype("uint8")
#     img0 = Image.fromarray(img0, 'RGB')
#     img0.show()