import os
import os.path as path
import csv
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class Logger(object):
    def __init__(self, log_dir, epoch=0, name="log"):
        self.log_dir = log_dir
        self.epoch = epoch
        self.name = name if name != "" else "log"
        self.iter_visual_freq = float('inf')
        self.loss_freq = float('inf')
        self.save_freq = float('inf')
        self.format_float = \
            lambda x: np.format_float_scientific(x, exp_digits=1, precision=2)

    def _to_dict(self, d):
        # TODO: two dicts pointing to each other triggers an infinite recursion
        if type(d) is defaultdict:
            d = dict(d)
        for k, v in d.items():
            if type(v) is dict or type(v) is defaultdict:
                d[k] = self._to_dict(v)
        return d

    def reset(self):
        if hasattr(self, 'loss'): self.loss = defaultdict(list)
        if hasattr(self, 'metrics'): self.metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def add_loss_log(self, loss_fcn, loss_freq, window_size=100):
        self.loss = defaultdict(list)
        self.loss_fcn = loss_fcn
        self.loss_freq = loss_freq
        self.window_size = window_size

    def add_save_log(self, save_fcn, save_freq):
        self.save_fcn = save_fcn
        self.save_freq = save_freq
        
        if hasattr(self.save_fcn, "__self__"):
            model = self.save_fcn.__self__
            with open(path.join(self.log_dir, "graph.txt"), "w") as f:
                f.write(self.get_graph(model))

    def add_eval_log(self, eval_fcn, eval_freq):
        self.eval_fcn = eval_fcn
        self.eval_freq = eval_freq

    def add_metric_log(self, pair_fcn, metrics_fcns, metrics_freq=1):
        self.pair_fcn = pair_fcn
        self.metrics_cnt = 0
        self.metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.metrics_fcns = metrics_fcns
        self.metrics_freq = metrics_freq

    def add_iter_visual_log(self, iter_visual_fcn, iter_visual_freq, name=""):
        self.iter_visual_fcn = iter_visual_fcn
        self.iter_visual_freq = iter_visual_freq
        self.iter_visual_name = name

    def add_epoch_visual_log(self, epoch_visual_fcn, epoch_visual_freq, name=""):
        self.epoch_visual_fcn = epoch_visual_fcn
        self.epoch_visual_freq = epoch_visual_freq
        self.epoch_visual_name = name

    def set_progress(self, progress):
        desc = '[{}][epoch{}]'.format(self.name, self.epoch)
        if hasattr(self, 'loss'):
            if len(self.loss) < 5:
                loss_str = " ".join(["{} {:.2e}({:.2e})".format(
                    k, v[-1], np.mean(v)) for k, v in self.loss.items()])
            else:
                loss_str = " ".join(["{} {}".format(
                    k, self.format_float(np.mean(v)))
                    for k, v in self.loss.items()])

            desc += loss_str
        if hasattr(self, 'metrics'):
            res_str = " "
            for k, res in self.metrics['mean'].items():
                res_str += "{}-> ".format(k)
                for j, m in res.items():
                    res_str += "{}: {:.2e} ".format(j, m)
                res_str += " "
            desc += res_str

        progress.set_description(desc=desc)

    def get_graph(self, model):
        model_str = ""
        if hasattr(model, 'parameters'):
            model_str += model.__repr__() + "\n"
        else:
            for k in model.__dir__():
                if not k.startswith("_"):
                    v = getattr(model, k)
                    if hasattr(v, 'parameters'):
                        model_str += k + ":\n"
                        model_str += self.get_graph(v)
        return model_str

    def __call__(self, iterable):
        progress = tqdm(iterable, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        for it, obj in enumerate(progress):
            yield obj

            if hasattr(self, 'loss_fcn') and it % self.loss_freq == 0:
                loss = self.loss_fcn()
                for k, v in loss.items():
                    if len(self.loss[k]) > self.window_size:
                        self.loss[k].pop(0)
                    self.loss[k].append(v)

                log_file = path.join(self.log_dir, 'loss.csv')
                with open(log_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.epoch, it] + list(loss.values()))

            if hasattr(self, 'iter_visual_fcn') and it % self.iter_visual_freq == 0:
                for k, v in self.iter_visual_fcn().items():
                    iter_visual_dir = path.join(self.log_dir, self.iter_visual_name)
                    if not path.isdir(iter_visual_dir): os.makedirs(iter_visual_dir)
                    visual_file = path.join(iter_visual_dir,
                        "epoch{}_iter{}_{}.png".format(self.epoch, it, k))
                    Image.fromarray(v).convert('RGB').save(visual_file)

            if hasattr(self, 'pair_fcn') and it % self.metrics_freq == self.metrics_freq - 1:
                pairs, name = self.pair_fcn()
                for i in range(len(pairs[0][1][0])):
                    n = len(self.metrics) - ('mean' in self.metrics)
                    for j, pair in pairs:
                        for k, metrics_fcn in self.metrics_fcns:
                            m = metrics_fcn(pair[0][i], pair[1][i]).tolist()
                            self.metrics[name[i] if name else n][j][k] = m
                            self.metrics['mean'][j][k] = (self.metrics['mean'][j][k] * n + m) / (n + 1)

                metric_file = path.join(self.log_dir, "metrics_{}.yaml".format(self.epoch))
                metrics_str = yaml.dump(self._to_dict(self.metrics), default_flow_style=False)
                with open(metric_file, 'w') as f: f.write(metrics_str)

            self.set_progress(progress)

        if hasattr(self, 'save_fcn') and \
           self.epoch % self.save_freq == self.save_freq - 1:  
            save_file = path.join(self.log_dir, "net_{}.pt".format(self.epoch))
            print("[Epoch {}] Saving {}".format(self.epoch, save_file))
            self.save_fcn(save_file)

        if hasattr(self, 'eval_fcn') and \
           self.epoch % self.eval_freq == self.eval_freq - 1:  
           self.eval_fcn()

        if hasattr(self, 'epoch_visual_fcn') and \
           self.epoch % self.epoch_visual_freq == self.epoch_visual_freq - 1:
            epoch_visual_dir = path.join(self.log_dir, self.epoch_visual_name)
            visual_dir = path.join(epoch_visual_dir, "epoch{}".format(self.epoch))
            if not path.isdir(visual_dir): os.makedirs(visual_dir)

            print("[Epoch {}] Evaluating...".format(self.epoch))
            for i, visuals in enumerate(self.epoch_visual_fcn()):
                for k, v in visuals.items():
                    visual_file = path.join(visual_dir,
                        "epoch{}_{}_{}.png".format(self.epoch, k, i))
                    Image.fromarray(v).convert('RGB').save(visual_file)
        self.epoch += 1