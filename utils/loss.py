import numpy
import torch
import torch.nn as nn
import torchvision.models as models


# borrow from http://arxiv.org/abs/2203.15836(VPTR) code
import torchsummary
import torchvision.models


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            target_tensor = target_tensor.to(prediction.device)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


# 根据论文中公式(6)实现
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.net = models.vgg16(pretrained=True)
        for _, param in self.net.named_parameters():
            param.requires_grad = False

    def get_features(self, net, inputs):
        """

        :param net: feature net
        :param inputs: input images, shape should be (N, C, H, W)
        :return: list of feature maps of each block of vgg16
        """
        result = []
        feats = inputs
        for i in range(31):
            if isinstance(net.features[i], torch.nn.ReLU):
                net.features[i] = torch.nn.ReLU(inplace=False)
                # torchvision.models中的模型结构，默认ReLU的inplace为True,会使得feats不断被更新，就无法得到想要那一层的输出了
            feats = net.features[i](feats)
            if i == 2 or i == 7 or i == 14 or i == 21 or i == 28:
                result.append(feats)
        return result

    def __call__(self, true_images, pred_images):
        """

        :param true_images:
        :param pred_images:
        :return:
        """
        assert len(true_images.shape) == 5  # (N,T,C,H,W)
        assert len(pred_images.shape) == 5
        true_feats = self.get_features(self.net, true_images.flatten(0, 1))  # [(N*T,C,H,W), (N*T,C1,H1,W1),...]
        pred_feats = self.get_features(self.net, pred_images.flatten(0, 1))
        loss_func = nn.MSELoss()
        loss = 0.
        for i in range(len(true_feats)):
            # 根据论文公式，对每个位置的元素，分母是分子的2范数，相当于分母是对分子取绝对值
            true_feats[i] = true_feats[i] / torch.abs(true_feats[i])
            pred_feats[i] = pred_feats[i] / torch.abs(pred_feats[i])
            loss += loss_func(true_feats[i], pred_feats[i])
        return loss
