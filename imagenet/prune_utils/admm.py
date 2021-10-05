from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import copy
from skimage.util.shape import view_as_windows

import time
import datetime
import operator
import random
from .prune_base import PruneBase
from .utils_pr import *


# from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc

# M:N pattern pruning
import heapq
from collections import defaultdict

admm = None


def prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Admm arguments')
    update_freq = admm_args.add_mutually_exclusive_group()
    update_freq.add_argument('--sp-admm-update-epoch', type=int,
                        help="how often we do admm update")
    update_freq.add_argument('--sp-admm-update-batch', type=int,
                        help="update admm after how many minibatches")
    admm_args.add_argument('--sp-admm-rho', type=float,
                        help="define rho for ADMM, overrides the rho specified in config file")
    admm_args.add_argument('--sp-admm-sparsity-type', type=str, default='irregular',
                        help="define sp_admm_sparsity_type: [irregular, irregular_global, column,filter]")
    admm_args.add_argument('--sp-admm-lr', type=float, default=0.001,
                        help="define learning rate for ADMM, reset to sp_admm_lr every time U,Z is updated. Overrides the learning rate of the outside training loop")
    admm_args.add_argument('--sp-global-weight-sparsity', type=float, default=-1, help="Use global weight magnitude to prune, override the --sp-config-file")
    admm_args.add_argument('--sp-predefine-global-weight-sparsity-dir', type=str, default=None,
                           help="define global sparsity based on a sparse model in this dir")



class ADMM(PruneBase):
    def __init__(self, args, model, logger=None, initialize=True):
        super(ADMM, self).__init__(args, model, logger)
        # this is to keep in CPU
        self.ADMM_U = {}
        self.ADMM_Z = {}
        # this is the identical copy in GPU. We do this separation
        # because in some cases in GPU run out of space if modified
        # directly
        self.ADMM_U_GPU = {}
        self.ADMM_Z_GPU = {}
        self.rhos = {}
        self.rho = None


        assert args.sp_config_file is not None, "Config file must be specified for ADMM"
        self.logger.info("Initializing ADMM pruning algorithm")

        if self.args.sp_admm_update_epoch is not None:
            self.update_epoch = self.args.sp_admm_update_epoch
        elif 'admm_update_epoch' in self.configs:
            self.update_epoch = self.configs["admm_update_epoch"]
        else:
            self.update_epoch = None
        if self.args.sp_admm_update_batch is not None:
            self.update_batch = self.args.sp_admm_update_batch
        elif 'admm_update_batch' in self.configs:
            self.update_batch = self.configs["admm_update_batch"]
        else:
            self.update_batch = None

        assert (self.update_epoch is None and self.update_batch is not None) or \
               (self.update_epoch is not None and self.update_batch is None)

        assert self.prune_ratios is not None
        if 'rho' in self.configs:
            self.rho = self.configs['rho']
        else:
            assert self.args.sp_admm_rho is not None
            self.rho = self.args.sp_admm_rho
        self.logger.info("ADMM rho is set to {}".format(str(self.rho)))

        if self.args.sp_load_prune_params is not None:
            self.prune_load_params()
        elif initialize:
            self.init()

    def init(self):
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            self.rhos[name] = self.rho
            prune_ratio = self.prune_ratios[name]

            self.logger.info("ADMM initialzing {}".format(name))
            updated_Z = self.prune_weight(name, W, prune_ratio, first)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her

            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U[name] = torch.zeros(W.shape).detach().cpu().float()
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

    def prune_harden(self, option=None):
        super(ADMM, self).prune_harden()

        if self.args.sp_global_weight_sparsity > 0:
            update_prune_ratio(self.args, self.model, self.prune_ratios, self.args.sp_global_weight_sparsity)

        for key in self.prune_ratios:
            print("prune_ratios[{}]:{}".format(key,self.prune_ratios[key]))

        #self.logger.info("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:  # ignore layers that do not have rho
                continue
            cuda_pruned_weights = None
            prune_ratio = self.prune_ratios[name]
            if option == None:
                cuda_pruned_weights = self.prune_weight(name, W, prune_ratio, first)  # get sparse model in cuda
                first = False
            W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

            non_zeros = W.detach().cpu().numpy() != 0
            non_zeros = non_zeros.astype(np.float32)
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
            #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))

    # this is expected to be in the very beginning of the epoch
    def prune_update(self, epoch=0, batch_idx=0):
        if ((self.update_epoch is not None) and ((epoch == 0) or \
            (epoch % self.update_epoch != 0))) or \
            ((self.update_batch is not None) and ((batch_idx == 0) or  \
             (batch_idx % self.update_batch != 0))) :
            return

        super(ADMM, self).prune_update(epoch, batch_idx)
        # this is to avoid the bug that GPU memory overflow
        for key in self.ADMM_Z:
            del self.ADMM_Z_GPU[key]
        for key in self.ADMM_U:
            del self.ADMM_U_GPU[key]
        first = True
        for i, (name, W) in enumerate(self.model.named_parameters()):
            if name not in self.prune_ratios:
                continue
            Z_prev = None
            W_CPU = W.detach().cpu().float()

            admm_z = W_CPU + self.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            updated_Z = self.prune_weight(name, admm_z, self.prune_ratios[name], first)  # equivalent to Euclidean Projection
            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()

            self.ADMM_U[name] = (W_CPU - self.ADMM_Z[name] + self.ADMM_U[name]).float()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

    def prune_update_combined_loss(self, ce_loss):
        admm_loss = {}
        for i, (name, W) in enumerate(self.model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in self.prune_ratios:
                continue
            if self.prune_ratios[name] == 0.0:
                continue
            admm_loss[name] = (0.5 * self.rhos[name] * \
                (torch.norm(W.float() - self.ADMM_Z_GPU[name].float() +
                self.ADMM_U_GPU[name].float(), p=2) ** 2)).float()

        total_admm_loss = 0
        for k, v in admm_loss.items():
            total_admm_loss += v
        mixed_loss = total_admm_loss + ce_loss

        return ce_loss, admm_loss, mixed_loss

    def prune_update_loss(self, ce_loss):
        _, _, combined_loss = self.prune_update_combined_loss(ce_loss)
        return combined_loss

    def prune_load_params(self):
        variables = self._prune_load_params()
        if variables == None:
            return
        self.logger.info("Loading ADMM variables")
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            if self.prune_ratios[name] == 0.0:
                continue
            cname = self._canonical_name(name)
            n = name if name in variables["U"] else cname
            if n not in variables["U"]:
                self.logger.warning("Param {} cannot be found in saved param file".format(n))
            self.ADMM_U[name] = variables["U"][n]
            self.ADMM_Z[name] = variables["Z"][n]
            self.rhos[name] = variables["rhos"][n]
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)

    def prune_store_params(self):
        if not self.args.sp_store_prune_params:
            return
        self.logger.info("Storing ADMM variables")
        variables = {
            "U": self.ADMM_U,
            "Z": self.ADMM_Z,
            "rhos": self.rhos,
        }
        self._prune_store_params(variables)

    def prune_weight(self, name, weight, prune_ratio, first):
        if prune_ratio == 0.0:
            return weight
        # if pruning too many items, just prune everything
        if prune_ratio >= 0.999999:
            return weight * 0.0
        if self.args.sp_admm_sparsity_type == "irregular_global":
            # res = self.weight_pruning_irregular_global(weight, prune_ratio, first)
            _, res = weight_pruning(self.args, self.configs, name, weight, prune_ratio)
        else:
            sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type)
            sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+")
            if len(sparsity_type_list) != 1: #multiple sparsity type
                print(sparsity_type_list)
                for i in range(len(sparsity_type_list)):
                    sparsity_type = sparsity_type_list[i]
                    print("* sparsity type {} is {}".format(i, sparsity_type))
                    self.args.sp_admm_sparsity_type = sparsity_type
                    _, weight =  weight_pruning(self.args, self.configs, name, weight, prune_ratio)
                    self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                    print(np.sum(weight.detach().cpu().numpy() != 0))
                return weight.to(weight.device).type(weight.dtype)
            else:
                _, res = weight_pruning(self.args, self.configs, name, weight, prune_ratio)


        return res.to(weight.device).type(weight.dtype)

    def weight_pruning_irregular_global(self, weight, prune_ratio, first):
        with torch.no_grad():
            if first:
                self.irregular_global_blob = None
                total_size = 0
                for i, (name, W) in enumerate(self.model.named_parameters()):
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    total_size += W.numel()
                to_prune = torch.zeros(total_size)
                index_ = 0
                for (name, W) in self.model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    size = W.numel()
                    to_prune[index_:(index_+size)] = W.view(-1).abs().clone()
                    index_ += size
                sorted_to_prune, _ = torch.sort(to_prune)
                self.irregular_global_blob = sorted_to_prune

            total_size = self.irregular_global_blob.numel()
            thre_index = int(total_size * prune_ratio)
            global_th = self.irregular_global_blob[thre_index]
            above_threshold = (weight.detach().cpu().float().abs() >
                global_th).to(weight.device).type(weight.dtype)
            weight = (weight * above_threshold).type(weight.dtype)
            return weight



def update_prune_ratio(args, model, prune_ratios, global_sparsity):
    if args.sp_predefine_global_weight_sparsity_dir is not None:
        # use layer sparsity in a predefined sparse model to override prune_ratios
        print("=> loading checkpoint for keep ratio: {}".format(args.sp_predefine_global_weight_sparsity_dir))

        assert os.path.exists(args.sp_predefine_global_weight_sparsity_dir), "\n\n * Error, pre_defined sparse mask model not exist!"

        checkpoint = torch.load(args.sp_predefine_global_weight_sparsity_dir, map_location="cuda")
        model_state = checkpoint["state_dict"]
        for name, weight in model_state.items():
            if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
                continue
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            non_zero = np.sum(weight.cpu().detach().numpy() != 0)
            new_prune_ratio = float(zeros / (zeros + non_zero))
            prune_ratios[name] = new_prune_ratio
        return prune_ratios

    total_size = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        total_size += W.data.numel()
    to_prune = np.zeros(total_size)
    index = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
        index += size
    #sorted_to_prune = np.sort(to_prune)
    threshold = np.percentile(to_prune, global_sparsity*100)

    # update prune_ratios key-value pairs
    total_zeros = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        np_W_abs = W.detach().cpu().abs().numpy()
        new_prune_ratio = float(np.sum(np_W_abs < threshold))/size
        if new_prune_ratio >= 0.999:
            new_prune_ratio = 0.99

        total_zeros += float(np.sum(np_W_abs < threshold))

        prune_ratios[name] = new_prune_ratio

    print("Updated prune_ratios:")
    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key,prune_ratios[key]))
    total_sparsity = total_zeros / total_size
    print("Total sparsity:{}".format(total_sparsity))

    return prune_ratios


def weight_pruning(args, configs, name, w, prune_ratio, mask_fixed_params=None):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """
    torch_weight = w
    weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy
    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    percent = prune_ratio * 100
    if (args.sp_admm_sparsity_type == "irregular") or (args.sp_admm_sparsity_type == "irregular_global"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        # weight[under_threshold] = 0
        ww = weight * above_threshold
        return torch.from_numpy(above_threshold), torch.from_numpy(ww)

    raise SyntaxError("Unknown sparsity type: {}".format(args.sp_admm_sparsity_type))




def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every args.sp_admm_update_epoch/3 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default
    admm epoch is 9)

    """
    admm_epoch = args.sp_admm_update_epoch
    lr = None

    if (epoch) % admm_epoch == 0:
        lr = args.sp_admm_lr
    else:
        admm_epoch_offset = (epoch) % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.sp_admm_lr * (0.1 ** (admm_epoch_offset // admm_step))

    #print(admm_epoch, args.sp_admm_lr, (epoch) % admm_epoch, lr)
    #input('?')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



