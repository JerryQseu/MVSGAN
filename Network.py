import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ExtraLayers import PixelShuffle3D


def crop_and_concat( upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

class Generator(nn.Module):
    def __init__(self,channel):
        super(Generator, self).__init__()

        self.saxmine1 = nn.utils.weight_norm(nn.Conv3d(channel,8,kernel_size=3,padding=1))
        self.lkrelu1 = nn.LeakyReLU(inplace=False)
        self.saxmine2 = nn.utils.weight_norm(nn.Conv3d(8, 16, kernel_size=3, padding=1))
        self.lkrelu2 = nn.LeakyReLU(inplace=False)
        self.saxmine3 = nn.utils.weight_norm(nn.Conv3d(16, 32, kernel_size=3, padding=1))
        self.lkrelu3 = nn.LeakyReLU(inplace=False)
        self.csaxup = nn.utils.weight_norm(nn.Conv3d(32,4,kernel_size=1))
        self.dsaxup = PixelShuffle3D(4)

        # for p in self.parameters():
        #     p.requires_grad = False


        # self.laxfuse1 = nn.utils.weight_norm(nn.Conv3d(3,8,kernel_size=3,padding=1))
        # self.laxfuse2 = nn.MaxPool3d((2, 1, 1))
        # self.laxfuse3 = nn.utils.weight_norm(nn.Conv3d(8, 3, kernel_size=1))

        self.saxlaxfuse = nn.utils.weight_norm(nn.Conv3d(4,8,kernel_size=3,padding=1))

        self.rdb1_conv1 = nn.utils.weight_norm((nn.Conv3d(8,16,kernel_size=3,padding=1)))
        self.rdb1_conv2 = nn.utils.weight_norm((nn.Conv3d(24, 32, kernel_size=3, padding=1)))
        self.rdb1_conv3 = nn.utils.weight_norm((nn.Conv3d(56, 64, kernel_size=3, padding=1)))
        self.lkrelu4 = nn.LeakyReLU(inplace=False)
        self.rdb1_cat = nn.Conv3d(120,8,kernel_size=1)

        self.rdb2_conv1 = nn.utils.weight_norm((nn.Conv3d(8, 16, kernel_size=3, padding=1)))
        self.rdb2_conv2 = nn.utils.weight_norm((nn.Conv3d(24, 32, kernel_size=3, padding=1)))
        self.rdb2_conv3 = nn.utils.weight_norm((nn.Conv3d(56, 64, kernel_size=3, padding=1)))
        self.lkrelu5 = nn.LeakyReLU(inplace=False)
        self.rdb2_cat = nn.Conv3d(120, 8, kernel_size=1)

        self.rdb3_conv1 = nn.utils.weight_norm((nn.Conv3d(8, 16, kernel_size=3, padding=1)))
        self.rdb3_conv2 = nn.utils.weight_norm((nn.Conv3d(24, 32, kernel_size=3, padding=1)))
        self.rdb3_conv3 = nn.utils.weight_norm((nn.Conv3d(56, 64, kernel_size=3, padding=1)))
        self.lkrelu6 = nn.LeakyReLU(inplace=False)
        self.rdb3_cat = nn.Conv3d(120, 8, kernel_size=1)

        self.cat = nn.Conv3d(24,32,kernel_size=1)
        self.fin = nn.utils.weight_norm(nn.Conv3d(32,64,kernel_size=3,padding=1))
        self.lkrelu7 = nn.LeakyReLU(inplace=False)
        self.fin1 = nn.utils.weight_norm(nn.Conv3d(64, 32, kernel_size=3, padding=1))
        self.fin2 = nn.utils.weight_norm(nn.Conv3d(32, 16, kernel_size=3, padding=1))
        self.lkrelu8 = nn.LeakyReLU(inplace=False)
        self.fin3 = nn.utils.weight_norm(nn.Conv3d(16,1,kernel_size=1))
        self.lkrelu9 = nn.LeakyReLU(inplace=False)

        self.fin4 = nn.Conv3d(1,1,kernel_size=1)

        self.suoxiao = nn.MaxPool3d((4,1,1))

    def forward(self, in_sax, in_2ch, in_3ch, in_4ch):
        saxmine = self.lkrelu3(self.saxmine3(self.lkrelu2(self.saxmine2(self.lkrelu1(self.saxmine1(in_sax))))))
        coarsesax1 = self.dsaxup(self.csaxup(saxmine))

        coarsesax = F.pad(coarsesax1,(0,0,0,0,0,32))
        # print(coarsesax.shape)

        # mm = in_2ch+in_3ch+in_4ch+coarsesax
        laxmix = torch.cat([in_2ch,in_3ch,in_4ch,coarsesax], dim=1)
        # laxmix = self.laxfuse3(self.laxfuse2(self.laxfuse1(laxmix)))

        # mix = torch.cat([mm,laxmix], dim=1)

        saxlaxfuse = self.saxlaxfuse(laxmix)
        ########
        rdb1_conv1 = self.rdb1_conv1(saxlaxfuse)
        rdb1_conv11 = torch.cat([saxlaxfuse,rdb1_conv1],dim=1)

        rdb1_conv2 = self.rdb1_conv2(rdb1_conv11)
        rdb1_conv22 = torch.cat([saxlaxfuse, rdb1_conv1,rdb1_conv2],dim=1)

        rdb1_conv3 = self.rdb1_conv3(rdb1_conv22)
        rdb1_conv3 = self.lkrelu4(rdb1_conv3)
        rdb1_conv33 = torch.cat([saxlaxfuse,rdb1_conv1,rdb1_conv2,rdb1_conv3],dim=1)

        rdb1_cat = self.rdb1_cat(rdb1_conv33)
        rdb1 = rdb1_cat+saxlaxfuse
        ########
        rdb2_conv1 = self.rdb2_conv1(rdb1)
        rdb2_conv11 = torch.cat([rdb1, rdb2_conv1], dim=1)

        rdb2_conv2 = self.rdb2_conv2(rdb2_conv11)
        rdb2_conv22 = torch.cat([rdb1, rdb2_conv1, rdb2_conv2], dim=1)

        rdb2_conv3 = self.rdb2_conv3(rdb2_conv22)
        rdb2_conv3 = self.lkrelu5(rdb2_conv3)
        rdb2_conv33 = torch.cat([rdb1, rdb2_conv1, rdb2_conv2, rdb2_conv3], dim=1)

        rdb2_cat = self.rdb2_cat(rdb2_conv33)
        rdb2 = rdb2_cat + rdb1
        ########
        rdb3_conv1 = self.rdb3_conv1(rdb2)
        rdb3_conv11 = torch.cat([rdb2, rdb3_conv1], dim=1)

        rdb3_conv2 = self.rdb3_conv2(rdb3_conv11)
        rdb3_conv22 = torch.cat([rdb2, rdb3_conv1, rdb3_conv2], dim=1)

        rdb3_conv3 = self.rdb3_conv3(rdb3_conv22)
        rdb3_conv3 = self.lkrelu6(rdb3_conv3)
        rdb3_conv33 = torch.cat([rdb2, rdb3_conv1, rdb3_conv2, rdb3_conv3], dim=1)

        rdb3_cat = self.rdb3_cat(rdb3_conv33)
        rdb3 = rdb3_cat + rdb2

        ########
        rdbs = torch.cat([rdb1,rdb2,rdb3],dim=1)
        cat = self.cat(rdbs)
        fin = self.lkrelu7(self.fin(cat))
        fin = self.fin1(fin)
        fin = self.lkrelu8(self.fin2(fin))
        fin = self.fin3(fin)
        fin = self.lkrelu9(fin)

        denssax = fin+coarsesax
        denssax = self.fin4(denssax)

        suoxiao = self.suoxiao(coarsesax1)
        suoxiao1 = suoxiao

        suoxiao2 = denssax[:,:,:64,:,:]
        suoxiao2 = self.suoxiao(suoxiao2)
        # suoxiao2 = torch.cat([suoxiao2[:,:,:10,:,:],suoxiao[:,:,10:,:,:]],dim=2)

        # suoxiao2[:, :, 0, :, :] = denssax[:, :, 0, :, :]
        # for i in range(1,16):
        #     suoxiao2[:, :, i, :, :] = denssax[:, :, i * 4, :, :]






        # coarse_suoxiao[:, :, 0, :, :] = coarsesax[:, :, 0, :, :]
        # # print(suoxiao.size())
        # for i in range(1, 24):
        #     coarse_suoxiao[:, :, i, :, :] = coarsesax[:, :, i * 4, :, :]


        out=[]
        out.append(denssax)#96*128*128
        out.append(coarsesax)#96*128*128
        out.append(suoxiao)#16*128*128
        out.append(suoxiao2)  # 16*128*128
        return out


class high_discriminator(nn.Module):
    def __init__(self,channel):
        super(high_discriminator, self).__init__()###128*128*64

        self.output_shape = (1,64, 128)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channel, 4, normalize=False),  ###64*64*28
            *discriminator_block(4, 8),  ###32*32*14
            *discriminator_block(8, 4),  ###16*16*7
            nn.Conv2d(4, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class low_discriminator(nn.Module):
    def __init__(self,channel):
        super(low_discriminator, self).__init__()###128*128*56

        self.output_shape = (1,128, 128)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channel, 4, normalize=False),  ###64*64*28
            *discriminator_block(4, 8),  ###32*32*14
            *discriminator_block(8, 4),  ###16*16*7
            nn.Conv2d(4, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class SegNet(nn.Module):
    def __init__(self, inchannel):
        super(SegNet, self).__init__()

        self.generator = Generator(channel=1)

        self.conv3D1 = nn.Conv3d(inchannel, 8, kernel_size=3, padding=1)#128*128*64
        self.bn3D1 = nn.BatchNorm3d(8)
        self.relu3D1 = nn.LeakyReLU(inplace=False)

        self.pool3D1 = nn.MaxPool3d(2)

        self.conv3D1_2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)#64*64*32
        self.bn3D1_2 = nn.BatchNorm3d(16)
        self.relu3D1_2 = nn.LeakyReLU(inplace=False)

        self.pool3D2 = nn.MaxPool3d(2)

        self.conv3D1_3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)#32*32*16
        self.bn3D1_3 = nn.BatchNorm3d(32)
        self.relu3D1_3 = nn.LeakyReLU(inplace=False)

        self.pool3D3 = nn.MaxPool3d(2)

        self.conv3D1_4 = nn.Conv3d(32,64, kernel_size=3, padding=1)#16*16*8
        self.bn3D1_4 = nn.BatchNorm3d(64)
        self.relu3D1_4 = nn.LeakyReLU(inplace=False)

        self.upool3D1 = nn.ConvTranspose3d(64,32,3,(2,2,2),1,(1,1,1))#32*32*16

        self.conv3D2_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.bn3D2_1 = nn.BatchNorm3d(32)
        self.relu3D2_1 = nn.LeakyReLU(inplace=False)

        self.upool3D2 = nn.ConvTranspose3d(32,16,3,(2,2,2),1,(1,1,1))#64*64*32

        self.conv3D2_2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.bn3D2_2 = nn.BatchNorm3d(16)
        self.relu3D2_2 = nn.LeakyReLU(inplace=False)

        self.upool3D3 = nn.ConvTranspose3d(16,8,3,(2,2,2),1,(1,1,1))#128*128*64

        self.conv3D2_3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout(0.7)
        self.bn3D2_3 = nn.BatchNorm3d(32)
        self.relu3D2_3 = nn.LeakyReLU(inplace=False)
        self.conv3D2_4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout(0.7)
        self.bn3D2_4 = nn.BatchNorm3d(64)
        self.relu3D2_4 = nn.LeakyReLU(inplace=False)
        self.conv3D2_5_1 = nn.Conv3d(64, 128, kernel_size=3,padding=1)
        self.conv3D2_5_2 = nn.Conv3d(128, 2, kernel_size=1)


        self.df1 = nn.AvgPool3d((4, 1, 1))#128*128*32



        self.sf = nn.Softmax(1)


    def forward(self, in_sax, in_2ch, in_3ch, in_4ch):
        out = []

        x = self.generator(in_sax, in_2ch, in_3ch, in_4ch)

        conv3d1_1 = self.relu3D1(self.bn3D1(self.conv3D1(x[0][:,:,:64,:,:])))
        conv3d1 = self.pool3D1(conv3d1_1)

        conv3d1_2_1 = self.relu3D1_2(self.bn3D1_2(self.conv3D1_2(conv3d1)))
        conv3d1_2 = self.pool3D2(conv3d1_2_1)

        conv3d1_3_1 = self.relu3D1_3(self.bn3D1_3(self.conv3D1_3(conv3d1_2)))
        conv3d1_3 = self.pool3D3(conv3d1_3_1)

        conv3d1_4_1 = self.relu3D1_4(self.bn3D1_4(self.conv3D1_4(conv3d1_3)))

        conv3d1_5 = self.upool3D1(conv3d1_4_1)

        conv3d1_5 = torch.cat((conv3d1_5,conv3d1_3_1),1)
        conv3d2_1 = self.relu3D2_1(self.bn3D2_1(self.conv3D2_1(conv3d1_5)))

        conv3d2_1 = self.upool3D2(conv3d2_1)

        # conv3d2_1 = F.pad(conv3d2_1, (0, 0, 0, 0,0,1))
        #print(bypass.shape)

        conv3d2_1 = torch.cat((conv3d2_1, conv3d1_2_1), 1)
        conv3d2_2 = self.relu3D2_2(self.bn3D2_2(self.conv3D2_2(conv3d2_1)))

        conv3d2_2 = self.upool3D3(conv3d2_2)

        conv3d2_2 = torch.cat((conv3d2_2, conv3d1_1), 1)
        conv3d2_3 = self.relu3D2_3(self.bn3D2_3(self.drop1(self.conv3D2_3(conv3d2_2))))
        conv3d2_4 = self.relu3D2_4(self.bn3D2_4(self.drop2(self.conv3D2_4(conv3d2_3))))


        conv3d2_4 = self.conv3D2_5_2(self.conv3D2_5_1(conv3d2_4))

        sf = self.sf(conv3d2_4)

        fin = self.df1(sf)


        fin[:, :, 0, :, :] = sf[:, :, 0, :, :]
        for i in range(1, 16):
            fin[:, :, i, :, :] = sf[:, :, i * 4, :, :]

        # sf_suoxiao = self.pool3D2_5_1(sf)

        # sf_suoxiao = torch.cat((x[2],x[2]),dim=1)
        # sf_suoxiao[:, :, 0, :, :] = sf[:, :, 0, :, :]
        # for i in range(1, 15):
        #     sf_suoxiao[:, :, i, :, :] = sf[:, :, i * 4, :, :]


        out.append(x[0])
        out.append(x[1])
        out.append(x[2])
        out.append(x[3])
        out.append(fin)
        return out









        
