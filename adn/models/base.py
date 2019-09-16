import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import types
from collections import OrderedDict
from torchvision.utils import make_grid
from ..utils import get_device, to_npy, backprop_on, backprop_off


class Base(nn.Module):
    def __init__(self, *opts):
        super(Base, self).__init__()
        self._get_opts(*opts)

    def _get_opts(self, *opts):
        for opt in opts:
            for k, v in opt.items(): setattr(self, k, v)

    def _match_device(self, *data):
        device = get_device(self)
        if len(data) == 1: return data[0].to(device)
        else: return (d.to(device) for d in data)

    def _get_visuals(self, lookup, n, func=None, normalize=True):
        if func is None: func = lambda x: x
        pairs = [(t, func(getattr(self, k)[:n])) for t, k in lookup if hasattr(self, k)]
        tags, images = zip(*pairs)
        tags, images = "_".join(tags), torch.cat(images)

        return {tags: self._make_visuals(images, len(pairs), normalize)}

    def _get_state_attrs(self):
        '''get the attributes with states (for saving and loading)
        '''
        state_attrs = {}
        for k in dir(self):
            if k[0] != '_':
                v = getattr(self, k)
                if hasattr(v, 'state_dict'):
                    state_attrs[k] = v
                    if hasattr(v, '_optimizer'):
                        state_attrs[k + ".optimizer"] = v._optimizer
                    if hasattr(v, '_scheduler'):
                        state_attrs[k + ".scheduler"] = v._scheduler
                    
        return state_attrs.items()

    def _make_visuals(self, images, n_rows, normalize=True, wide=True):
        if normalize: images = (images - images.min()) / (images.max() - images.min())
        visuals = make_grid(images, nrow=images.shape[0] // n_rows, normalize=False)
        visuals = to_npy(visuals).transpose(1, 2, 0)
        visuals = (visuals * 255).astype(np.uint8)
        # if wide and visuals.shape[0] > visuals.shape[1]:
        #     visuals = visuals.transpose(1, 0, 2)
        return visuals

    def forward(self, *data): raise NotImplementedError

    def get_visuals(self): raise NotImplementedError

    def resume(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        for k, v in self._get_state_attrs():
            if k in checkpoint: v.load_state_dict(checkpoint[k])


class BaseTrain(Base):
    def __init__(self, *opts):
        super(BaseTrain, self).__init__(*opts)

    def _get_trainer(self, model_dict, model_type, opt_type="adam"):
        model = model_dict[model_type]()
        optimizer, scheduler = self._setup_learn(model.parameters(), opt_type)

        def _clear(self, backprop=True):
            """ Clear gradient
            """
            if backprop: backprop_on(self)
            self.zero_grad()
            self._loss = OrderedDict()

        def _update(self, backprop=True):
            """ Backward errors and update weights
            """
            if len(self._loss) == 0: return

            loss = sum(self._loss.values())
            self._loss['all'] = loss

            loss.backward()
            self._optimizer.step()
            if not backprop: backprop_off(self)

        class CriterionDict(object):
            def __init__(self, obj):
                self.obj = obj
                self.dict = {}
            
            def __setitem__(self, key, item):
                self.dict[key] = types.MethodType(item, self.obj)
            
            def __getitem__(self, key): return self.dict[key]

        model.__dict__['_optimizer'] = optimizer
        model.__dict__['_scheduler'] = scheduler
        model.__dict__['_loss'] = OrderedDict()
        model.__dict__['_criterion'] = CriterionDict(model)
        model.__dict__['_update'] = types.MethodType(_update, model)
        model.__dict__['_clear'] = types.MethodType(_clear, model)
        return model

    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt += p[1]
        return wgt

    def _setup_learn(self, params, opt_type="adam"):
        """ Setup optimizer and learning rate scheduler
        """
        if opt_type == "adam":
            optimizer = optim.Adam(params, lr=self.lr,
                betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                step_size=self.step_size, gamma=self.gamma)
            return optimizer, scheduler

    def _get_criterion(self, loss_dict, loss_wgts, tag=""):
        if type(loss_wgts) is list:
            if type(loss_wgts[0]) is str: loss_wgts = dict([loss_wgts])
            else: loss_wgts = dict(loss_wgts)

        def criterion(self, *inputs):
            for t, v in loss_wgts.items():
                if v == 0.0: continue
                if callable(loss_dict[t]): # normal loss functions
                    loss_fcn = loss_dict[t]()
                    self._loss[tag + t] = v * loss_fcn(*inputs)
                else: # GAN loss functions
                    g_fcn, d_fcn = loss_dict[t]
                    fake, real = inputs

                    self._loss[tag + t + "_g"] = g_fcn(fake, real) # loss for generator
                    d_fcn.__self__._loss[tag + t + "_d"] = d_fcn(fake.detach(), real) # loss for discriminator
        return criterion

    def _summarize_loss(self, losses, sum_name="all"):
        """ Collect loss from different sources and return collected loss with the sum
        """
        def summarize_loss(losses, prefix=""):
            total_loss = {}
            if type(losses) is dict:
                for name, loss in losses.items():
                    loss_k = summarize_loss(
                        loss, (prefix + "_" if prefix else prefix) + name)
                    for k, v in loss_k.items(): total_loss[k] = v
            else: total_loss[prefix] = losses
            return total_loss

        losses = summarize_loss(losses)
        if len(losses) > 1: losses[sum_name] = sum(losses.values())
        else: losses[sum_name] = list(losses.values())[0]
        return losses

    def _merge_loss(self, *losses):
        losses = [(loss._loss if hasattr(loss, '_loss') else loss) for loss in losses]
        losses = sum([list(loss.items()) for loss in losses], [])
        loss = OrderedDict(losses)

        self.loss = loss
        self.loss.pop('all')
        return loss

    def _clear(self, *models):
        for model in models: model._clear()

    def optimize(self, *data): raise NotImplementedError

    def save(self, checkpoint_file):
        checkpoint = {k: v.state_dict() for k, v in self._get_state_attrs()}
        torch.save(checkpoint, checkpoint_file)

    def get_loss(self): # define self.loss first
        if len(self.loss) == 2 and 'all' in self.loss: self.loss.pop('all')
        return {t: to_npy(v) for t, v in self.loss.items()}

    def update_lr(self):
        for k in dir(self):
            if k.startswith('_'): continue
            v = getattr(self, k)
            if 'lr_scheduler' in str(type(v)): v.step()
            if hasattr(v, '_scheduler'): v._scheduler.step()
