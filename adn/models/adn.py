import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import ADN, NLayerDiscriminator, add_gan_loss
from ..utils import print_model, get_device


class ADNTrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADNTrain, self).__init__(learn_opts, loss_opts)
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
              adn = lambda: ADN(**g_opts),
            nlayer = lambda: NLayerDiscriminator(**d_opts))

        self.model_g = self._get_trainer(model_dict, g_type) # ADN generators
        self.model_dl = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for low quality image (with artifact)
        self.model_dh = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for high quality image (without artifact)

        loss_dict = dict(
               l1 = nn.L1Loss,
               gl = (self.model_dl.get_g_loss, self.model_dl.get_d_loss), # GAN loss for low quality image.
               gh = (self.model_dh.get_g_loss, self.model_dh.get_d_loss)) # GAN loss for high quality image

        # Create criterion for different loss types
        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["art"] = self._get_criterion(loss_dict, self.wgts["art"], "art_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])

        print_model(self)
    
    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt += p[1]
        return wgt

    def optimize(self, img_low, img_high):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.model_g._clear()

        # low -> low_l, low -> low_h
        if self._nonzero_weight("gl", "lh", "ll"):
            self.model_dl._clear()
            self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
            self.model_g._criterion["gl"](self.pred_lh, self.img_high)
            self.model_g._criterion["lh"](self.pred_lh, self.img_high)
            self.model_g._criterion["ll"](self.pred_ll, self.img_low)
            self.model_dl._update()

        # high -> high_l, high -> high_h
        if self._nonzero_weight("gh", "hh"):
            self.model_dh._clear()
            self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
            self.model_g._criterion["gh"](self.pred_hl, self.img_low)
            self.model_g._criterion["hh"](self.pred_hh, self.img_high)
            self.model_dh._update()

        # low_h -> low_h_l
        if self._nonzero_weight("lhl"):
            self.pred_lhl = self.model_g.forward_hl(self.pred_hl, self.pred_lh)
            self.model_g._criterion["lhl"](self.pred_lhl, self.img_low)

        # high_l -> high_l_h
        if self._nonzero_weight("hlh"):
            self.pred_hlh = self.model_g.forward_lh(self.pred_hl)
            self.model_g._criterion["hlh"](self.pred_hlh, self.img_high)

        # artifact
        if self._nonzero_weight("art"):
            ll = self.img_low if self.gt_art else self.pred_ll
            hh = self.img_high if self.gt_art else self.pred_hh
            self.model_g._criterion["art"](
                ll - self.pred_lh, self.pred_hl - hh)

        self.model_g._update()

        # merge losses for printing
        self.loss = self._merge_loss(
            self.model_dl._loss, self.model_dh._loss, self.model_g._loss)

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"), ("lhl", "pred_lhl"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh"), ("hlh", "pred_hlh")]

        return self._get_visuals(lookup, n)

    def evaluate(self, loader, metrics):
        progress = tqdm(loader)
        res = defaultdict(lambda: defaultdict(float))
        cnt = 0
        for img_low, img_high in progress:
            img_low, img_high = self._match_device(img_low, img_high)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data

            pred_ll, pred_lh = self.model_g.forward1(img_low)
            pred_hl, pred_hh = self.model_g.forward2(img_low, img_high)
            pred_hlh = self.model_g.forward_lh(pred_hl)
            img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh)

            met = {
                "ll": metrics(img_low, pred_ll),
                "lh": metrics(img_high, pred_lh),
                "hl": metrics(img_low, pred_hl),
                "hh": metrics(img_high, pred_hh),
                "hlh":metrics(img_high, pred_hlh)}

            res = {n: {k: (res[n][k] * cnt + v) / (cnt + 1) for k, v in met[n].items()} for n in met}
            desc = "[{}]".format("/".join(met["ll"].keys()))
            for n, met in res.items():
                vals = "/".join(("{:.2f}".format(v) for v in met.values()))
                desc += " {}: {}".format(n, vals)
            progress.set_description(desc=desc)


class ADNTest(Base):
    def __init__(self, g_type, **model_opts):
        super(ADNTest, self).__init__()

        g_opts = model_opts[g_type]
        model_dict = dict(adn = lambda: ADN(**g_opts))
        self.model_g = model_dict[g_type]()
        print_model(self)

    def forward(self, img_low):
        self.img_low = self._match_device(img_low)
        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)

        return  self.pred_ll, self.pred_lh

    def evaluate(self, img_low, img_high, name=None):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.name = name

        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
        self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
        self.pred_hlh = self.model_g.forward_lh(self.pred_hl)

    def get_pairs(self):
        return [
            ("before", (self.img_low, self.img_high)), 
            ("after", (self.pred_lh, self.img_high))], self.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh")]
        func = lambda x: x * 0.5 + 0.5
        return self._get_visuals(lookup, n, func, False)
