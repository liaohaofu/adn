import types
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from torch.nn import functional as F
from ..utils import get_device


def add_gan_loss(model, gan_type="lsgan"):
    """ Add GAN loss to a discriminator (model)
    """

    if hasattr(model, "get_d_loss") and hasattr(model, "get_g_loss"): return model

    gan_loss = GANLoss(gan_type)

    def get_d_loss(self, fake, real):
        """ Get the loss that updates the discriminator
        """

        nonlocal gan_loss

        device = get_device(self)
        gan_loss = gan_loss.to(device)

        pred_real = self(real)
        pred_fake = self(fake.detach())

        loss_fake = gan_loss(pred_fake, False)
        loss_real = gan_loss(pred_real, True)

        loss = (loss_real + loss_fake) * 0.5
        return loss

    def get_g_loss(self, fake, real):
        """ Get the loss that updates the generator
        """

        nonlocal gan_loss

        device = get_device(self)
        gan_loss = gan_loss.to(device)

        pred_fake = self(fake)
        loss = gan_loss(pred_fake, True)

        return loss

    model.__dict__["get_d_loss"] = types.MethodType(get_d_loss, model)
    model.__dict__["get_g_loss"] = types.MethodType(get_g_loss, model)
    return model


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
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
            prediction (tensor) - - tpyically the prediction from a discriminator
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
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
