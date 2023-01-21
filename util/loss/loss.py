import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

__all__ = ['MyWeightedL1Loss', 'SegmentationLosses', 'MyLossJC']


class MyWeightedL1Loss(nn.L1Loss):
    def __init__(self, reduction='none'):
        super(MyWeightedL1Loss, self).__init__(reduction=reduction)

    def forward(self, input, target, pixel_weight):
        pixel_mse = super(MyWeightedL1Loss, self).forward(input, target)
        loss = pixel_mse * pixel_weight
        return loss.sum()/(loss.size(0)*1000)
    

class MyLossJC(nn.Module):
    def __init__(self, n_classes):
        super(MyLossJC, self).__init__()
        self.n_classes = n_classes

    def forward(self, seg_softmax, target_seg_train):

        pha_v = torch.zeros((target_seg_train.shape[0], self.n_classes, target_seg_train.shape[1], target_seg_train.shape[2]), device=target_seg_train.device)
        for ind_batch in range(target_seg_train.shape[0]):
            for label_v in range(self.n_classes):
                label_loc = target_seg_train[ind_batch].eq(label_v)
                num_label = label_loc.sum()
                if not num_label.eq(0):
                    pha_v[ind_batch][label_v][label_loc] = torch.tensor(1.0, device=target_seg_train.device) / num_label

        loss_JC = torch.tensor(0.0, requires_grad=True, device=seg_softmax.device)
        for ind_batch in range(seg_softmax.shape[0]):
            for label_positive in range(self.n_classes):
                z_i = seg_softmax[ind_batch][label_positive]
                pha_i = pha_v[ind_batch][label_positive]
                alpha_i = (z_i * pha_i).sum()
                for label_negative in range(self.n_classes):
                    beta_ik = ((1 - z_i) * pha_v[ind_batch][label_negative]).sum()
                    loss_JC = loss_JC + (0.5*(alpha_i + beta_ik + np.spacing(1))).log()
        loss_JC = -0.5 * loss_JC/seg_softmax.shape[0]
        return loss_JC


class SegmentationLosses(nn.CrossEntropyLoss):
    def __init__(self, name='dice_loss', se_loss=False,
                 aux_weight=None, weight=None, ignore_index=0):
        '''2D Cross Entropy Loss with Auxiliary Loss or Dice Loss

        :param name: (string) type of loss : ['dice_loss', 'cross_entropy', 'cross_entropy_with_dice']
        :param aux_weight: (float) weights_map of an auxiliary layer or the weight of dice loss
        :param weight: (torch.tensor) the weights_map of each class
        :param ignore_index: (torch.tensor) ignore i class.
        '''
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.name = name
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = True
        self.reduce = True
        print('Using loss: {}'.format(name))

    def forward(self, *inputs):
        if self.name == 'dice_loss':
            return self._dice_loss2(*inputs) #self._dice_loss(*inputs)
        elif self.name == 'cross_entropy':
            if self.aux_weight == 0 or self.aux_weight is None:
                return super(SegmentationLosses, self).forward(*inputs)
            else:
                pred1, pred2 = inputs[0]
                target = inputs[1]
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                return loss1 + self.aux_weight * loss2
        elif self.name == 'cross_entropy_with_dice':
            return super(SegmentationLosses, self).forward(*inputs)\
                   + self.aux_weight * self._dice_loss2(*inputs)
        else:
            raise NotImplementedError

    def _dice_loss1(self, input, target):
        """
        input : (NxCxHxW Tensor) which is feature output as output = model_G(x)
        target :  (NxHxW LongTensor)
        :return: the average dice loss for each channel
        """
        smooth = 1.0

        probs = F.softmax(input, dim=1)
        encoded_target = probs.detach() * 0

        # one-hot encoding
        if self.ignore_index != -1:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if self.weight is None:
            weight = 1

        intersection = probs * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = probs + encoded_target

        if self.ignore_index != -1:
            denominator[mask] = 0

        denominator = denominator.sum(0).sum(1).sum(1)
        loss_per_channel = weight * (1 - ((numerator + smooth) / (denominator + smooth)))
        average = encoded_target.size()[0] if self.reduction == 'mean' else 1

        return loss_per_channel.mean().mean()

    def _dice_loss2(self, input, target, optimize_bg=False, smooth=1.0):
        """input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        def dice_coefficient(input, target, smooth=1.0):

            assert smooth > 0, 'Smooth must be greater than 0.'
            probs = F.softmax(input, dim=1)

            encoded_target = probs.detach() * 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            encoded_target = encoded_target.float()

            num = probs * encoded_target  # b, c, h, w -- p*g
            num = torch.sum(num, dim=3)  # b, c, h
            num = torch.sum(num, dim=2)  # b, c

            den1 = probs * probs  # b, c, h, w -- p^2
            den1 = torch.sum(den1, dim=3)  # b, c, h
            den1 = torch.sum(den1, dim=2)  # b, c

            den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
            den2 = torch.sum(den2, dim=3)  # b, c, h
            den2 = torch.sum(den2, dim=2)  # b, c

            dice = (2 * num + smooth) / (den1 + den2 + smooth)  # b, c

            return dice

        dice = dice_coefficient(input, target, smooth=smooth)

        if not optimize_bg:
            dice = dice[:, 1:]                 # we ignore bg dice val, and take the fg

        if not type(self.weight) is type(None):
            if not optimize_bg:
                weight = self.weight[1:]             # ignore bg weight
            weight = weight.size(0) * weight / weight.sum()  # normalize fg weights_map
            dice = dice * weight                # weighting

        dice_loss = 1 - dice.mean(1)     # loss is calculated using mean over dice vals (n,c) -> (n)

        if not self.reduce:
            return dice_loss

        if self.size_average:
            return dice_loss.mean()

        return dice_loss.sum()



