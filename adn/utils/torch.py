"""Helper functions for torch
"""

__all__ = [
    "get_device", "is_cuda", "copy_model", "find_layer", "to_npy", "get_last_checkpoint",
    "print_model", "save_graph", "backprop_on", "backprop_off", "add_post", "flatten_model",
    "FunctionModel"]

import os
import os.path as path
import numpy as np
import torch
import torch.nn as nn
from copy import copy
from .misc import read_dir
from collections import OrderedDict


def get_device(model):
    return next(model.parameters()).device


def is_cuda(model):
    return next(model.parameters()).is_cuda


def copy_model(model):
    """shallow copy a model
    """
    if len(list(model.children())) == 0: return model

    model_ = copy(model)
    model_._modules = copy(model_._modules)
    for k, m in model._modules.items():
        model_._modules[k] = copy_model(m)
    return model_


def find_layer(module, filter_fcn):
    def find_layer_(module, found):
        for k, m in module.named_children():
            if filter_fcn(m): found.append((module, k))
            else: find_layer_(m, found)
    found = []
    find_layer_(module, found)
    return found


class FunctionModel(nn.Module):
    def __init__(self, fcn):
        super(FunctionModel, self).__init__()
        self.fcn = fcn

    def forward(self, *inputs):
        return self.fcn(*inputs)


def to_npy(*tensors, squeeze=False):
    if len(tensors) == 1:
        if squeeze: return tensors[0].detach().cpu().numpy().squeeze()
        else: return tensors[0].detach().cpu().numpy()
    else:
        if squeeze: return [t.detach().cpu().numpy().squeeze() for t in tensors]
        else: return [t.detach().cpu().numpy() for t in tensors]


def set_requires_grad(*nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def backprop_on(*nets): set_requires_grad(*nets, requires_grad=True)


def backprop_off(*nets): set_requires_grad(*nets, requires_grad=False)


def get_last_checkpoint(checkpoint_dir, predicate=None, pattern=None):
    if predicate is None:
        predicate = lambda x: x.endswith('pth') or x.endswith('pt')
    
    checkpoints = read_dir(checkpoint_dir, predicate)
    if len(checkpoints) == 0:
        return "", 0
    checkpoints = sorted(checkpoints, key=lambda x: path.getmtime(x))
    
    checkpoint = checkpoints[-1]
    if pattern is None:
        pattern = lambda x: int(path.basename(x).split('_')[-1].split('.')[0])
    return checkpoint, pattern(checkpoint)


def print_model(model): print(get_graph(model))


def save_graph(model, graph_file):
    with open(graph_file, 'w') as f: f.write(get_graph(model))


def get_graph(model):
    def get_graph_(model, param_cnts):
        model_str = ""
        if hasattr(model, 'parameters'):
            model_str += model.__repr__() + "\n"
            parameters = [p for p in model.parameters() if p.requires_grad]
            num_parameters = sum([np.prod(p.size()) for p in parameters])
            param_cnts.append(num_parameters)
        else:
            for k in model.__dir__():
                if not k.startswith("_"):
                    v = getattr(model, k)
                    if hasattr(v, 'parameters'):
                        model_str += k + ":\n"
                        model_str += get_graph_(v, param_cnts)
        return model_str

    model_str = ""
    param_cnts = []
    model_str += '============ Model Initialized ============\n'
    model_str += get_graph_(model, param_cnts)
    model_str += '===========================================\n'
    model_str += "Number of parameters: {:.4e}\n".format(sum(param_cnts))
    return model_str


def add_post(loader, post_fcn):
    class LoaderWrapper(object):
        def __init__(self, loader, post_fcn):
            self.loader = loader
            self.post_fcn = post_fcn
        
        def __getattribute__(self, name):
            if not name.startswith("__") and name not in object.__getattribute__(self, "__dict__") :
                return getattr(object.__getattribute__(self, "loader"), name)
            return object.__getattribute__(self, name)

        def __len__(self): return len(self.loader)

        def __iter__(self):
            for data in self.loader:
                yield self.post_fcn(data)

    return LoaderWrapper(loader, post_fcn)


def flatten_model(model):
    def flatten_model_(model, output):
        model_list = list(model.children())
        if len(model_list) == 1: model = model_list[0]

        if type(model) is nn.Sequential:
            for m in model.children():
                flatten_model_(m, output)
        else:
            output.append(model)

    output = []
    flatten_model_(model, output)
    return nn.Sequential(*output)
