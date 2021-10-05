import numpy as np
import torch
import copy
import logging
import os
import sys

from . import utils_pr
from .sparse_train import SparseTraining, prune_parse_arguments as retrain_parse_arguments

from .utils_pr import load_configs, canonical_name
from .utils_pr import weight_pruning, weight_growing


sparse_training = None

def main_prune_parse_arguments(parser):
    parser.add_argument('--sp-admm', action='store_true', default=False,
                        help="for admm pruning")
    parser.add_argument('--sp-admm-update-epoch', type=int,
                        help="how often we do admm update")
    parser.add_argument('--sp-admm-lr', type=float, default=0.001,
                        help="define learning rate for ADMM, reset to sp_admm_lr every time U,Z is updated. Overrides the learning rate of the outside training loop")
    parser.add_argument('--sp-retrain', action='store_true',
                        help="Retrain a pruned model")
    parser.add_argument('--sp-config-file', type=str,
                        help="define config file")
    parser.add_argument('--sp-admm-sparsity-type', type=str, default='irregular_global',
                        help="define sp_admm_sparsity_type: [irregular, irregular_global]")
    parser.add_argument('--sp-global-weight-sparsity', type=float, default=-1,
                        help="Use global weight magnitude to prune, override the --sp-config-file")

class PruneBase(object):
    def __init__(self, args, model, logger=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.configs = None
        self.prune_ratios = None

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s',
                                level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger)

        if args.sp_global_weight_sparsity > 0:
            update_prune_ratio(model, self.prune_ratios, args.sp_global_weight_sparsity)


    def prune_harden(self):
        self.logger.info("Hard prune")

    def prune_update(self, epoch=0, batch_idx=0):
        self.logger.info("Update prune, epoch={}, batch={}".\
            format(epoch, batch_idx))

    def prune_update_loss(self, loss):
        pass

    def prune_update_combined_loss(self, loss):
        pass

    def apply_masks(self):
        pass



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

        assert self.prune_ratios is not None
        if 'rho' in self.configs:
            self.rho = self.configs['rho']
        else:
            assert self.args.sp_admm_rho is not None
            self.rho = self.args.sp_admm_rho
        self.logger.info("ADMM rho is set to {}".format(str(self.rho)))

        if initialize:
            self.init()

    def init(self):
        first = True

        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            self.rhos[name] = self.rho
            prune_ratio = self.prune_ratios[name]
            print(name, prune_ratio)


            self.logger.info("ADMM initialzing {}".format(name))
            updated_Z = prune_weight(self.args, self.configs, name, W, prune_ratio, first) # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her

            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U[name] = torch.zeros(W.shape).detach().cpu().float()
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)


    def prune_update(self, epoch=0, batch_idx=0):
        if ((self.update_epoch is not None) and ((epoch == 0) or (epoch % self.update_epoch != 0))):
            return
        if batch_idx != 0:
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

            updated_Z = prune_weight(self.args, self.configs, name, admm_z, self.prune_ratios[name], first) # equivalent to Euclidean Projection
            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()

            self.ADMM_U[name] = (W_CPU - self.ADMM_Z[name] + self.ADMM_U[name]).float()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)



def prune_parse_arguments(parser):
    main_prune_parse_arguments(parser)
    retrain_parse_arguments(parser)


def prune_init(args, model, logger=None, pre_defined_mask=None):
    global prune_algo, sparse_training

    prune_algo = None
    sparse_training = None

    if args.sp_retrain:
        if args.sp_prune_before_retrain:
            prune_harden(args, model)
        sparse_training = SparseTraining(args, model, logger, pre_defined_mask)
        return

    if args.sp_admm:
        prune_algo = ADMM(args, model, logger)
        return


def prune_update(epoch=0, batch_idx=0):
    if prune_algo != None:
        return prune_algo.prune_update(epoch, batch_idx)
    elif sparse_training != None:
        sparse_training.update_mask(epoch, batch_idx)


def prune_harden(args, model, option=None):
    configs, prune_ratios = load_configs(model, args.sp_config_file, logger=None)

    if args.sp_global_weight_sparsity > 0:
        update_prune_ratio(model, prune_ratios, args.sp_global_weight_sparsity)

    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key, prune_ratios[key]))

    # self.logger.info("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
    print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
    first = True
    for (name, W) in model.named_parameters():
        if name not in prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        prune_ratio = prune_ratios[name]
        if option == None:
            cuda_pruned_weights = prune_weight(args, configs, name, W, prune_ratio, first)  # get sparse model in cuda
            first = False
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

        if args.sp_admm_sparsity_type == "block":
            block = eval(args.sp_admm_block)
            if block[1] == -1:  # row pruning, need to delete corresponding bias
                bias_layer = name.replace(".weight", ".bias")
                with torch.no_grad():
                    bias = model.state_dict()[bias_layer]
                    bias_mask = torch.sum(W, 1)
                    bias_mask[bias_mask != 0] = 1
                    bias.mul_(bias_mask)
        elif args.sp_admm_sparsity_type == "filter":
            if not "downsample" in name:
                bn_weight_name = name.replace("conv", "bn")
                bn_bias_name = bn_weight_name.replace("weight", "bias")
            else:
                bn_weight_name = name.replace("downsample.0", "downsample.1")
                bn_bias_name = bn_weight_name.replace("weight", "bias")

            print("removing bn {}, {}".format(bn_weight_name, bn_bias_name))
            # bias_layer_name = name.replace(".weight", ".bias")

            with torch.no_grad():
                bn_weight = model.state_dict()[bn_weight_name]
                bn_bias = model.state_dict()[bn_bias_name]
                # bias = self.model.state_dict()[bias_layer_name]

                mask = torch.sum(W, (1, 2, 3))
                mask[mask != 0] = 1
                bn_weight.mul_(mask)
                bn_bias.mul_(mask)
                # bias.data.mul_(mask)

        non_zeros = W.detach().cpu().numpy() != 0
        non_zeros = non_zeros.astype(np.float32)
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
        # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))


def prune_weight(args, configs, name, weight, prune_ratio, first):
    if prune_ratio == 0.0:
        return weight
    # if pruning too many items, just prune everything
    if prune_ratio >= 0.99999:
        return weight * 0.0
    if args.sp_admm_sparsity_type == "irregular_global":
        _, res = weight_pruning(args, configs, name, weight, prune_ratio)
    else:
        sp_admm_sparsity_type_copy = copy.copy(args.sp_admm_sparsity_type)
        sparsity_type_list = (args.sp_admm_sparsity_type).split("+")
        if len(sparsity_type_list) != 1: #multiple sparsity type
            print(sparsity_type_list)
            for i in range(len(sparsity_type_list)):
                sparsity_type = sparsity_type_list[i]
                print("* sparsity type {} is {}".format(i, sparsity_type))
                args.sp_admm_sparsity_type = sparsity_type
                _, weight =  weight_pruning(args, configs, name, weight, prune_ratio)
                args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                print(np.sum(weight.detach().cpu().numpy() != 0))
            return weight.to(weight.device).type(weight.dtype)
        else:
            _, res = weight_pruning(args, configs, name, weight, prune_ratio)


    return res.to(weight.device).type(weight.dtype)



def prune_apply_masks():
    if sparse_training:
        sparse_training.apply_masks()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads():
    if sparse_training:
        sparse_training.apply_masks_on_grads()
    else:
        return
        assert(False)


def update_prune_ratio(model, prune_ratios, global_sparsity):
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


def prune_print_sparsity(model=None, logger=None, show_sparse_only=False, compressed_view=False):
    if model is None:
        if sparse_training:
            model = sparse_training.model
        else:
            return
    if logger:
        p = logger.info
    elif sparse_training:
        p = sparse_training.logger.info
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
    for (name, W) in model.named_parameters():
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))


def prune_update_learning_rate(optimizer, epoch, args):
    if prune_algo == None:
        return None
    return admm_adjust_learning_rate(optimizer, epoch, args)

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

