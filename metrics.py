import torch.nn as nn
import torch.nn.functional as F
import torch


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        inter = torch.sum(logits[:, 1, :, :, :] * targets[:, 1, :, :, :])
        union = torch.sum(logits[:, 1, :, :, :]) + torch.sum(targets[:, 1, :, :, :])
        dice = (2. * inter + 1) / (union + 1)
        dice_sum = dice_sum +dice*0.5
        # inter = torch.sum(logits[:, 2, :, :, :] * targets[:, 2, :, :, :])
        # union = torch.sum(logits[:, 2, :, :, :]) + torch.sum(targets[:, 2, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # dice_sum = dice_sum +dice*0.7
        # inter = torch.sum(logits[:, 3, :, :, :] * targets[:, 3, :, :, :])
        # union = torch.sum(logits[:, 3, :, :, :]) + torch.sum(targets[:, 3, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # dice_sum = dice_sum +dice*0.1
        # inter = torch.sum(logits[:, 4, :, :, :] * targets[:, 4, :, :, :])
        # union = torch.sum(logits[:, 4, :, :, :]) + torch.sum(targets[:, 4, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # dice_sum = dice_sum +dice*0.1
        inter = torch.sum(logits[:, 0, :, :, :] * targets[:, 0, :, :, :])
        union = torch.sum(logits[:, 0, :, :, :]) + torch.sum(targets[:, 0, :, :, :])
        dice = (2. * inter + 1) / (union + 1)
        dice_sum = dice_sum +dice*0.5

        # for i in range(class_num):
        #     inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
        #     union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
        #     dice = (2. * inter + 1) / (union + 1)
        #     dice_sum += dice
        # return 1 - dice_sum / class_num
        return 1 - dice_sum


class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):

        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)

        return 1 - 2. * inter / union


def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


def T(logits, targets):
    return torch.sum(targets[:, 2, :, :, :])


def P(logits, targets):
    return torch.sum(logits[:, 2, :, :, :])


def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :, :] * logits[:, 2, :, :, :])

class stdMeanLoss(nn.Module):
    def __init__(self):
        super(stdMeanLoss, self).__init__()

    def forward(self, logits, targets):
        a = logits.view(1, -1)
        b = targets.view(1, -1)
        a = F.normalize(a)
        b = F.normalize(b)
        loss = a.mm(b.t())
        # loss = loss.mul(loss)



        # for i in range(class_num):
        #     inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
        #     union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
        #     dice = (2. * inter + 1) / (union + 1)
        #     dice_sum += dice
        # return 1 - dice_sum / class_num
        return loss

from torch.autograd import Variable
import random
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))