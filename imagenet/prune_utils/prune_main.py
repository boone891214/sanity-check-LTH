import numpy as np
import argparse
import torch

from .prune_base import prune_parse_arguments as prune_base_parse_arguments
from .admm import ADMM, prune_parse_arguments as admm_prune_parse_arguments
from .retrain import Retrain, prune_parse_arguments as retrain_parse_arguments
from .admm import admm_adjust_learning_rate

from .utils_pr import prune_parse_arguments as utils_prune_parse_arguments

prune_algo = None
retrain = None


def prune_parse_arguments(parser):
    prune_base_parse_arguments(parser)
    admm_prune_parse_arguments(parser)
    utils_prune_parse_arguments(parser)
    retrain_parse_arguments(parser)


def prune_init(args, model, logger=None, fixed_model=None, pre_defined_mask=None):
    global prune_algo, retrain

    if args.sp_retrain:
        if args.sp_prune_before_retrain:
            # For prune before retrain, we need to also set sp-admm-sparsity-type in the command line
            # We need to set sp_admm_update_epoch for ADMM, so set it to 1.
            args.sp_admm_update_epoch = 1

            prune_algo = ADMM(args, model, logger)
            prune_algo.prune_harden()


        prune_algo = None
        retrain = Retrain(args, model, logger, pre_defined_mask)
        return

    if args.sp_admm:
        prune_algo = ADMM(args, model, logger)
        return

def prune_update(epoch=0, batch_idx=0):
    if prune_algo != None:
        return prune_algo.prune_update(epoch, batch_idx)

def prune_update_loss(loss):
    if prune_algo == None:
        return loss
    return prune_algo.prune_update_loss(loss)


def prune_update_combined_loss(loss):
    if prune_algo == None:
        return loss, loss, loss
    return prune_algo.prune_update_combined_loss(loss)



def prune_harden():
    if prune_algo == None:
        return None
    return prune_algo.prune_harden()


def prune_apply_masks():
    if retrain:
        retrain.apply_masks()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads():
    if retrain:
        retrain.apply_masks_on_grads()
    else:
        return
        assert(False)


def prune_print_sparsity(model=None, logger=None, show_sparse_only=False, compressed_view=False):
    if model is None:
        if prune_algo:
            model = prune_algo.model
        elif retrain:
            model = retrain.model
        else:
            return
    if logger:
        p = logger.info
    elif prune_algo:
        p = prune_algo.logger.info
    elif retrain:
        p = retrain.logger.info
    else:
        p = print

    if show_sparse_only:
        print("The sparsity of all params (>0.01): num_nonzeros, total_num, sparsity")
        for (name, W) in model.named_parameters():
            non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            if sparsity > 0.01:
                print("{}, {}, {}, {}, {}".format(name, non_zeros.shape, num_nonzeros, total_num, sparsity))
        return

    if compressed_view is True:
        total_w_num = 0
        total_w_num_nz = 0
        for (name, W) in model.named_parameters():
            if "weight" in name:
                non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                num_nonzeros = np.count_nonzero(non_zeros)
                total_w_num_nz += num_nonzeros
                total_num = non_zeros.size
                total_w_num += total_num

        sparsity = 1 - (total_w_num_nz * 1.0) / total_w_num
        print("The sparsity of all params with 'weights' in its name: num_nonzeros, total_num, sparsity")
        print("{}, {}, {}".format(total_w_num_nz, total_w_num, sparsity))
        return

    print("The sparsity of all parameters: name, num_nonzeros, total_num, shape, sparsity")
    total_size = 0
    total_zeros = 0
    for (name, W) in model.named_parameters():
        total_size += W.data.numel()
        total_zeros += np.sum(W.cpu().detach().numpy() == 0)
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))

    print("\nTotal sparsity: {}\n\n".format(total_zeros / total_size))


def prune_update_learning_rate(optimizer, epoch, args):
    if prune_algo == None:
        return None
    return admm_adjust_learning_rate(optimizer, epoch, args)


def prune_generate_yaml(model, sparsity, yaml_filename=None):
    if yaml_filename is None:
        yaml_filename = 'sp_{}.yaml'.format(sparsity)
    with open(yaml_filename,'w') as f:
        f.write("prune_ratios: \n")
    num_w = 0
    for name, W in model.named_parameters():
        print(name, W.shape)
        num_w += W.detach().cpu().numpy().size
        if len(W.detach().cpu().numpy().shape) > 1:
            with open(yaml_filename,'a') as f:
                if 'module.' in name:
                    f.write("{}: {}\n".format(name[7:], sparsity))
                else:
                    f.write("{}: {}\n".format(name, sparsity))
    print("Yaml file {} generated".format(yaml_filename))
    print("Total number of parameters: {}".format(num_w))
    exit()
