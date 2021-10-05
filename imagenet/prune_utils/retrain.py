
import torch
import logging
import sys
import os
import numpy as np
import argparse
import time
import random
import copy
from . import utils_pr
from .admm import weight_pruning, ADMM

def prune_parse_arguments(parser):
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight',
                    help="retrain mask pattern")
    parser.add_argument('--sp-prune-before-retrain', action='store_true',
                        help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None,
                        help="using another sparse model to init sparse mask")


class Retrain(object):
    def __init__(self, args, model, logger=None, pre_defined_mask=None, seed=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask # as model's state_dict

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.masks = {}
        self.masked_layers = {}
        self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger)

        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None


        self.init()

    def init(self):
        self.generate_mask(self.pre_defined_mask)


    def apply_masks(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    W.mul_((self.masks[name] != 0).type(dtype))
                    # W.data = (W * (self.masks[name] != 0).type(dtype)).type(dtype)
                    pass

    def apply_masks_on_grads(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
                    pass


    def generate_mask(self, pre_defined_mask=None):
        masks = {}
        # import pdb; pdb.set_trace()
        if self.pattern == 'weight':
            with torch.no_grad():
                for name, W in (self.model.named_parameters()):

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        elif self.pattern == "pre_defined":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue
                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.001:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask
        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
