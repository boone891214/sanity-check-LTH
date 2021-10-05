from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import copy
import yaml
import numpy as np

import datetime
import operator
import random

def prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Multi level admm arguments')
    admm_args.add_argument('--sp-load-frozen-weights',
        type=str, help='the weights that are frozen '
        'throughout the pruning process')


def canonical_name(name):
    # if the model is running in parallel, the name may start
    # with "module.", but if hte model is running in a single
    # GPU, it may not, we always filter the name to be the version
    # without "module.",
    # names in the config should not start with "module."
    if "module." in name:
        return name.replace("module.", "")
    else:
        return name


def _collect_dir_keys(configs, dir):
    if not isinstance(configs, dict):
        return

    for name in configs:
        if name not in dir:
            dir[name] = []
        dir[name].append(configs)
    for name in configs:
        _collect_dir_keys(configs[name], dir)


def _canonicalize_names(configs, model, logger):
    dir = {}
    collected_keys = _collect_dir_keys(configs, dir)
    for name in model.state_dict():
        cname = canonical_name(name)
        if cname == name:
            continue
        if name in dir:
            assert cname not in dir
            for parent in dir[name]:
                assert cname not in parent
                parent[cname] = parent[name]
                del parent[name]
            logger.info("Updating parameter from {} to {}".format(name, cname))


def load_configs(model, filename, logger):
    assert filename is not None, \
            "Config file must be specified"

    with open(filename, "r") as stream:
        try:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    _canonicalize_names(configs, model, logger)

    if "prune_ratios" in configs:
        config_prune_ratios = configs["prune_ratios"]

        count = 0
        prune_ratios = {}
        for name in model.state_dict():
            W = model.state_dict()[name]
            cname = canonical_name(name)

            if cname not in config_prune_ratios:
                continue
            count = count + 1
            prune_ratios[name] = config_prune_ratios[cname]
            if name != cname:
                logger.info("Map weight config name from {} to {}".\
                    format(cname, name))

        if len(prune_ratios) != len(config_prune_ratios):
            extra_weights = set(config_prune_ratios) - set(prune_ratios)
            for name in extra_weights:
                logger.warning("{} in config file cannot be found".\
                    format(name))


    return configs, prune_ratios



