import torch
import logging
import sys
import numpy as np
import copy
from . import utils_pr
from .utils_pr import weight_pruning, weight_growing

def prune_parse_arguments(parser):
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight',
                        help="retrain mask pattern")
    parser.add_argument('--sp-update-init-method', type=str, default='zero',
                        help="mask update initialization method")
    parser.add_argument('--sp-mask-update-freq', type=int, default=5,
                        help="how many epochs to update sparse mask")
    parser.add_argument('--retrain-mask-seed', type=int, default=None,
                    help="seed to generate a random mask")
    parser.add_argument('--sp-prune-before-retrain', action='store_true',
                        help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None,
                        help="using another sparse model to init sparse mask")


class SparseTraining(object):
    def __init__(self, args, model, logger=None, pre_defined_mask=None, seed=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask # as model's state_dict
        self.seed = self.args.retrain_mask_seed
        self.sp_mask_update_freq = self.args.sp_mask_update_freq
        self.update_init_method = self.args.sp_update_init_method

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

        if "upper_bound" in self.configs:
            self.upper_bound = self.configs['upper_bound']
        else:
            self.upper_bound = None
        if "lower_bound" in self.configs:
            self.lower_bound = self.configs['lower_bound']
        else:
            self.lower_bound = None
        if "mask_update_decay_epoch" in self.configs:
            self.mask_update_decay_epoch = self.configs['mask_update_decay_epoch']
        else:
            self.mask_update_decay_epoch = None

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

    def update_mask(self, epoch, batch_idx):
        # a hacky way to differenate random GaP and others
        if not self.mask_update_decay_epoch:
            return
        if batch_idx != 0:
            return

        freq = self.sp_mask_update_freq

        bound_index = 0

        # check yaml file and make necessary fix
        try: # if mask_update_decay_epoch has only one entry
            int(self.mask_update_decay_epoch)
            freq_decay_epoch = int(self.mask_update_decay_epoch)
            try: # if upper/lower bound have only one entry
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError: # if upper/lower bound have multiple entries
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity
                if epoch >= freq_decay_epoch:
                    freq *= 1
                    bound_index += 1
        except ValueError: # if mask_update_decay_epoch has multiple entries
            freq_decay_epoch = self.mask_update_decay_epoch.split('-')
            for i in range(len(freq_decay_epoch)):
                freq_decay_epoch[i] = int(freq_decay_epoch[i])

            try:
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError:
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity

                if len(freq_decay_epoch) + 1 <= len(upper_bound): # upper/lower bound num entries enough for all update
                    for decay in freq_decay_epoch:
                        if epoch >= decay:
                            freq *= 1
                            bound_index += 1
                else: # upper/lower bound num entries less than update needs, use the last entry to do rest updates
                    for idx, _ in enumerate(upper_bound):
                        if epoch >= freq_decay_epoch[idx] and idx != len(upper_bound) - 1:
                            freq *= 1
                            bound_index += 1

        lower_bound_value = float(lower_bound[bound_index])
        upper_bound_value = float(upper_bound[bound_index])

        if epoch % freq == 0:
            '''
            calculate prune_part and grow_part,
            set prune_part and grow_part to all layer specified in yaml file as random GaP do.
            '''
            prune_part, grow_part = self.model_partition()

            with torch.no_grad():
                for name, W in (self.model.named_parameters()):
                    if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                        continue

                    weight = W.cpu().detach().numpy()
                    weight_current_copy = copy.copy(weight)


                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    np_orig_mask = self.masks[name].cpu().detach().numpy()

                    print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,
                                                                    str(num_nonzeros),
                                                                    str(total_num),
                                                                    str(sparsity))))

                    ############## pruning #############
                    pruned_weight_np = None
                    if name in prune_part:
                        sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type)
                        sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+")
                        for i in range(len(sparsity_type_list)):
                            sparsity_type = sparsity_type_list[i]
                            print("* sparsity type {} is {}".format(i, sparsity_type))
                            self.args.sp_admm_sparsity_type = sparsity_type

                            pruned_mask, pruned_weight = weight_pruning(self.args,
                                                                        self.configs,
                                                                        name,
                                                                        W,
                                                                        lower_bound_value)
                            self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                            # pruned_mask_np = pruned_mask.cpu().detach().numpy()
                            pruned_weight_np = pruned_weight.cpu().detach().numpy()

                            W.mul_(pruned_mask.cuda())


                            non_zeros_prune = pruned_weight_np != 0
                            num_nonzeros_prune = np.count_nonzero(non_zeros_prune.astype(np.float32))
                            print(("==> PRUNE: {}: {}, {}, {}".format(name,
                                                             str(num_nonzeros_prune),
                                                             str(total_num),
                                                             str(1 - (num_nonzeros_prune * 1.0) / total_num))))

                            self.masks[name] = pruned_mask.cuda()


                    ############## growing #############
                    if name in grow_part:
                        if pruned_weight_np is None: # use in seq gap
                            pruned_weight_np = weight_current_copy

                        updated_mask = weight_growing(self.args,
                                                      name,
                                                      pruned_weight_np,
                                                      lower_bound_value,
                                                      upper_bound_value,
                                                      self.update_init_method)
                        self.masks[name] = updated_mask
                        pass


    def model_partition(self):
        prune_part = []
        grow_part = []

        for name, _ in self.model.named_parameters():
            if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                continue
            prune_part.append(name)
            grow_part.append(name)

        return prune_part, grow_part

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
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
