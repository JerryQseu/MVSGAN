import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
# import test_data as Data
import data_utils as Data
import torch.utils.data as Datas
import Network as Network
import math
import torch.nn.functional as F
import metrics as metrics
from torch.autograd import Variable




device = torch.device("cuda:0")
data = Data.dataset3
dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)

Gen = Network.SegNet(inchannel=1).to(device)
Dis1 = Network.high_discriminator(channel=1).to(device)
# Dis2 = Network.low_discriminator(channel=1).to(device)

fake_A_buffer = metrics.ReplayBuffer()
fake_B_buffer = metrics.ReplayBuffer()

# ###
pretrained_dict = torch.load('../pklh/generattor_epoch_78Network.pkl')
model_dict = Gen.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
Gen.load_state_dict(model_dict)
####
# pretrained_dict = torch.load('../pklh/Dis1_epoch_81Network.pkl')
# model_dict = Dis1.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Dis1.load_state_dict(model_dict)
####
# pretrained_dict = torch.load('../pkl/Dis2_epoch_7Network.pkl')
# model_dict = Dis2.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Dis2.load_state_dict(model_dict)



criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()

opt_gen = torch.optim.RMSprop(Gen.parameters(),lr=0.0001)
opt_dis1 = torch.optim.RMSprop(Dis1.parameters(),lr=0.0001)
# opt_dis2 = torch.optim.RMSprop(Dis2.parameters(),lr=0.00001)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


for epoch in range(200):
    for step, (sax_img, two_image, three_image, four_image, seg_labels) in enumerate(dataloder):
        flag = 1
        sax_img = sax_img.to(device).float()
        two_image = two_image.to(device).float()
        three_image = three_image.to(device).float()
        four_image = four_image.to(device).float()
        seg_labels = seg_labels.to(device).float()
        # dense_sax_img = dense_sax_img.to(device).float()
        # dense_sax_img2 = dense_sax_img2.to(device).float()
        # dense_sax_img[:, :, 48:, :] = 0
        # print(dense_sax_img.size())

        valid1 = Variable(Tensor(np.ones((1, *Dis1.output_shape))), requires_grad=False)
        fake1 = Variable(Tensor(np.zeros((1, *Dis1.output_shape))), requires_grad=False)
        # valid2 = Variable(Tensor(np.ones((1, *Dis2.output_shape))), requires_grad=False)
        # fake2 = Variable(Tensor(np.zeros((1, *Dis2.output_shape))), requires_grad=False)

        #####Seg Training
        for p in Dis1.parameters():  # reset requires_grad
            p.requires_grad = False   # they are set to False below in netG update
        # for p in Dis2.parameters():  # reset requires_grad
        #     p.requires_grad = False   # they are set to False below in netG update
        opt_gen.zero_grad()

        gen_out = Gen(sax_img, two_image, three_image, four_image)

        dense_sax = gen_out[0]
        coarse_dense_sax = gen_out[1]
        coarse_suoxiao = gen_out[2]
        suoxiao = gen_out[3]
        seg_result = gen_out[4]
        # print(seg_labels.shape)

        lax1 = dense_sax[:,:,:,:,64]
        lax2 = dense_sax[:, :, :, 64, :]

        loss_seg = metrics.DiceMeanLoss()(seg_result, seg_labels)
        loss_suoxiao = criterion_L1(suoxiao, sax_img)
        # loss_suoxiao1 = criterion_L1(coarse_suoxiao, sax_img)
        dz = torch.abs(dense_sax[:, :, 1:, :, :] - dense_sax[:, :, :-1, :, :])
        dx = torch.abs(dense_sax[:, :, :, 1:, :] - dense_sax[:, :, :, :-1, :])
        dy = torch.abs(dense_sax[:, :, :, :, 1:] - dense_sax[:, :, :, :, :-1])
        # lossflow = (torch.mean(dz.mul(dz))+torch.mean(dx.mul(dx))+torch.mean(dy.mul(dy)))/3
        lossflow = torch.mean(dz.mul(dz))

        dz1 = torch.abs(coarse_dense_sax[:, :, 1:, :, :] - coarse_dense_sax[:, :, :-1, :, :])
        lossflow1 = torch.mean(dz1.mul(dz1))

        loss_GAN_1_2 = criterion_MSE(Dis1(dense_sax[:,:,:64,:,64]), fake1)
        loss_GAN_1_4 = criterion_MSE(Dis1(dense_sax[:, :, :64, 64, :]), fake1)
        loss_GAN_2_2 = criterion_MSE(Dis1(dense_sax[:,:,:64,:,64]), valid1)
        loss_GAN_2_4 = criterion_MSE(Dis1(dense_sax[:, :, :64, 64, :]), valid1)
        # loss_GAN_1 = (loss_GAN_1_2+loss_GAN_1_4)*0.5
        # loss_GAN_2 = (loss_GAN_2_2+loss_GAN_2_4)*0.5
        # loss_GAN_1 = (10*loss_GAN_1+loss_GAN_2)*0.5
        # if loss_GAN_1 >=0.5:
        #
        #     if loss_GAN_2 >=0.5:
        #         loss_GAN_1 = loss_GAN_1 - 0.5
        #         loss_GAN_2 = loss_GAN_2-0.5
        #     else:
        #         loss_GAN_2 = loss_GAN_2+0.5
        #         loss_GAN_1 = loss_GAN_1 - 0.5
        # else:
        #
        #     if loss_GAN_2 >=0.5:
        #         loss_GAN_1 = loss_GAN_1 + 0.5
        #         loss_GAN_2 = loss_GAN_2-0.5
        #     else:
        #         loss_GAN_2 = loss_GAN_2+0.5
        #         loss_GAN_1 = loss_GAN_1 + 0.5

        loss_GAN_1 = (loss_GAN_1_2+loss_GAN_1_4+loss_GAN_2_2+loss_GAN_2_4)/4


        # loss_Lax = criterion_L1(lax1[:,:,60:,:], dense_sax_img[:,:,60:,:])
        # loss_Lax2 = criterion_L1(lax1, dense_sax_img_mirror)
        # loss_Lax = min(loss_Lax1,loss_Lax2)
        # loss_Lax = (1+ncc_loss(lax1, dense_sax_img)+1+ncc_loss(lax2, dense_sax_img2))/8
        # print(1+ncc_loss(dense_sax_img2, dense_sax_img2))

        # loss_GAN_lax1 = criterion_MSE(Dis2(lax1), valid2)
        # loss_GAN_lax2 = criterion_MSE(Dis2(lax2), valid2)
        # loss_GAN_lax = loss_GAN_lax1
        # if loss_GAN_2>1:
        #     loss_GAN_2 = loss_GAN_2-loss_GAN_2
        #loss_GAN = torch.abs(loss_GAN_1-0.5)#+loss_GAN_lax


        # loss_g =20*lossflow + 80*loss_suoxiao+10*loss_seg+80*loss_loss_GANLax+
        # = 4*lossflow + 5*loss_suoxiao + 5*loss_suoxiao1 + 5*loss_seg + 1*loss_Lax + loss_GAN
        loss_g = 5*lossflow + 5*loss_suoxiao + 1*loss_seg + loss_GAN_1+5*lossflow1
        # loss_g = 20 * loss_suoxiao + 10 * loss_seg + 10 * loss_Lax + loss_GAN####withoutFLOW
        # loss_g = 80 * lossflow + 20 * loss_suoxiao + 10 * loss_seg + 10 * loss_Lax###withoutGAN
        # loss_g = 80 * lossflow  + 10 * loss_seg + 10 * loss_Lax + loss_GAN###withoutsuoxiao
        # loss_g = 80 * lossflow + 20 * loss_suoxiao + 10 * loss_seg + loss_GAN####withou lax

        loss_g.backward(retain_graph=True)
        opt_gen.step()


        if epoch % 1 == 0:
            ######High Dis
            for p in Dis1.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            opt_dis1.zero_grad()

            loss_real = criterion_MSE(Dis1(coarse_dense_sax[:, :, :64, :, 64]), valid1)
            loss_two = criterion_MSE(Dis1(two_image[:, :, :64, :, 64]), fake1)
            loss_three = criterion_MSE(Dis1(three_image[:, :, :64, :, 64]), fake1)
            loss_four = criterion_MSE(Dis1(four_image[:, :, :64, :, 64]), fake1)

            loss_dis1 = (loss_real + (loss_two+loss_three+loss_four)/3)/2
            # fake_A_ = fake_A_buffer.push_and_pop(dense_sax[:, :, :64, :, :])
            # loss_fake = criterion_MSE(Dis1(fake_A_.detach()), valid1)
            # loss_dis1 = (loss_real + loss_fake) / 2
            loss_dis1.backward()
            opt_dis1.step()





        #
        # #######Low Dis
        # for p in Dis2.parameters():  # reset requires_grad
        #     p.requires_grad = True  # they are set to False below in netG update
        # opt_dis2.zero_grad()
        #
        #
        # loss_real = criterion_MSE(Dis2(dense_sax_img), valid2)
        # fake_B_ = fake_B_buffer.push_and_pop(lax1)
        # loss_fake1 = criterion_MSE(Dis2(fake_B_.detach()), fake2)
        # # fake_B_ = fake_B_buffer.push_and_pop(lax2)
        # # loss_fake2 = criterion_MSE(Dis2(fake_B_.detach()), fake2)
        # loss_dis2 = (loss_real + loss_fake1) / 2
        # if loss_dis2<0.1:
        #     loss_dis2 = loss_dis2+1
        # loss_dis2.backward()
        # opt_dis2.step()


    #####Log
        if step % 2 == 0:
            torch.save(Gen.state_dict(), '../pklh/generattor_epoch_' + str(epoch) + 'Network.pkl')
            torch.save(Dis1.state_dict(), '../pklh/Dis1_epoch_' + str(epoch) + 'Network.pkl')
            # torch.save(Dis2.state_dict(), '../pkl/Dis2_epoch_' + str(epoch) + 'Network.pkl')

        if step % 1 == 0:
            print('EPOCH:', epoch, '|Step:', step, '|loss_seg: %.3e' % loss_seg.data.cpu().numpy(), '|loss_suoxiao: %.3e' % loss_suoxiao.data.cpu().numpy(),  '|lossflow: %.3e' % lossflow1.data.cpu().numpy(), '|loss_GAN: %.3e' % loss_GAN_1.data.cpu().numpy(), '|loss_dis: %.3e' % loss_dis1.data.cpu().numpy())
       # torch.cuda.empty_cache()

        if step % 1 == 0:
            with torch.no_grad():
                pt = dense_sax[0, 0, :, :, :].data.cpu().numpy()
                # pt = np.transpose(pt, (2, 1, 0))
                out = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out, './constructed_dense_sax'+str(step)+'.nii')

                pt = coarse_dense_sax[0, 0, :, :, :].data.cpu().numpy()
                # pt = np.transpose(pt, (2, 1, 0))
                out = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out, './constructed_coarse_sax'+str(step)+'.nii')

                mm = seg_result[0, 1, :, :, :] * 1
                # pt = np.transpose(mm.data.cpu().numpy(), (2, 1, 0))
                out = sitk.GetImageFromArray(mm.data.cpu().numpy())
                sitk.WriteImage(out, './seg_result'+str(step)+'.nii')

                mm = seg_labels[0, 1, :, :, :] * 1
                # pt = np.transpose(mm.data.cpu().numpy(), (2, 1, 0))
                out = sitk.GetImageFromArray(mm.data.cpu().numpy())
                sitk.WriteImage(out, './label'+str(step)+'.nii')

                pt = sax_img[0, 0, :, :, :].data.cpu().numpy()
                # pt = np.transpose(pt, (2, 1, 0))
                out = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out, './sax'+str(step)+'.nii')

                # pt = dense_sax_img[0, 0,  :, :].data.cpu().numpy()
                # pt = np.transpose(pt, (2, 1, 0))
                # out = sitk.GetImageFromArray(pt)
                # sitk.WriteImage(out, './dense_sax.nii.gz')

                # pt = coarse_suoxiao[0, 0, :, :,:].data.cpu().numpy()
                # # pt = np.transpose(pt, (2, 1, 0))
                # out = sitk.GetImageFromArray(pt)
                # sitk.WriteImage(out, './lax1.nii.gz')

                pt = two_image[0, 0, :64, :,64].data.cpu().numpy()
                # pt = np.transpose(pt, (2, 1, 0))
                out = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out, './lax1'+str(step)+'.nii')









