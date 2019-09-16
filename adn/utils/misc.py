__all__ = ["read_dir", "get_config", "update_config", "save_config",
    "convert_coefficient2hu", "convert_hu2coefficient", "arange", "get_connected_components",
    "EasyDict"]

import os
import os.path as path
import yaml
import numpy as np


class EasyDict(object):
    def __init__(self, opt): self.opt = opt

    def __getattribute__(self, name):
        if name == 'opt' or name.startswith("_") or name not in self.opt:
            return object.__getattribute__(self, name)
        else: return self.opt[name]

    def __setattr__(self, name, value):
        if name == 'opt': object.__setattr__(self, name, value)
        else: self.opt[name] = value

    def __getitem__(self, name):
        return self.opt[name]
    
    def __setitem__(self, name, value):
        self.opt[name] = value

    def __contains__(self, item):
        return item in self.opt

    def __repr__(self):
        return self.opt.__repr__()

    def keys(self):
        return self.opt.keys()

    def values(self):
        return self.opt.values()

    def items(self):
        return self.opt.items()


def resolve_expression(config):
    if type(config) is dict:
        new_config = {}
        for k, v in config.items():
            if type(v) is str and v.startswith("!!python"):
                v = eval(v[8:])
            elif type(v) is dict:
                v = resolve_expression(v)
            new_config[k] = v
        config = new_config
    return config


def get_config(config_file, config_names=[]):
    ''' load config from file
    '''

    with open(config_file) as f:
        config = resolve_expression(yaml.load(f, Loader=yaml.FullLoader))
    
    if type(config_names) == str: return EasyDict(config[config_names])

    while len(config_names) != 0:
        config_name = config_names.pop(0)
        if config_name not in config:
            raise ValueError("Invalid config name: {}".format(config_name))
        config = config[config_name]

    return EasyDict(config)


def update_config(config, args):
    ''' rewrite default config with user input
    '''
    if args is None: return
    if hasattr(args, "__dict__"): args = args.__dict__
    for arg, val in args.items():
        # if not (val is None or val is False) and arg in config: config[arg] = val
        # TODO: this may cause bugs for other programs
        if arg in config: config[arg] = val
    
    for _, val in config.items():
        if type(val) == dict: update_config(val, args)


def save_config(config, config_file, print_opts=True):
    config_str = yaml.dump(config, default_flow_style=False)
    with open(config_file, 'w') as f: f.write(config_str)
    print('================= Options =================')
    print(config_str[:-1])
    print('===========================================')


def read_dir(dir_path, predicate=None, name_only=False, recursive=False):
    if type(predicate) is str:
        if predicate in {'dir', 'file'}:
            predicate = {
                'dir': lambda x: path.isdir(path.join(dir_path, x)),
                'file':lambda x: path.isfile(path.join(dir_path, x))
            }[predicate]
        else:
            ext = predicate
            predicate = lambda x: ext in path.splitext(x)[-1]
    elif type(predicate) is list:
        exts = predicate
        predicate = lambda x: path.splitext(x)[-1][1:] in exts

    def read_dir_(output, dir_path, predicate, name_only, recursive):
        if not path.isdir(dir_path): return
        for f in os.listdir(dir_path):
            d = path.join(dir_path, f)
            if predicate is None or predicate(f):
                output.append(f if name_only else d)
            if recursive and path.isdir(d):
                read_dir_(output, d, predicate, name_only, recursive)

    output = []
    read_dir_(output, dir_path, predicate, name_only, recursive)
    return sorted(output)


def convert_coefficient2hu(image):
    image = (image - 0.192) / 0.192 * 1000
    return image


def convert_hu2coefficient(image):
    image = image * 0.192 / 1000 + 0.192
    return image


def arange(start, stop, step):
    """ Matlab-like arange
    """
    r = np.arange(start, stop, step).tolist()
    if r[-1] + step == stop:
        r.append(stop)
    return np.array(r)


def get_connected_components(points):
    def get_neighbors(point):
        p0, p1 = point
        neighbors = [
            (p0 - 1, p1 - 1), (p0 - 1, p1), (p0 - 1, p1 + 1),
            (p0 + 1, p1 - 1), (p0 + 1, p1), (p0 + 1, p1 + 1),
            (p0, p1 - 1), (p0, p1 + 1)]
        return neighbors

    components = []
    while points:
        component = []
        unchecked = [points.pop()]
        while unchecked:
            point = unchecked.pop(0)
            neighbors = get_neighbors(point)
            for n in neighbors:
                if n in points:
                    points.remove(n)
                    unchecked.append(n)
            component.append(point)
        components.append(component)
    return components