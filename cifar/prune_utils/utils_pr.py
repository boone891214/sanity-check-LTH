from __future__ import print_function
import torch
import copy
import yaml
import numpy as np


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
            print("Updating parameter from {} to {}".format(name, cname))


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
                print("Map weight config name from {} to {}".\
                    format(cname, name))

        if len(prune_ratios) != len(config_prune_ratios):
            extra_weights = set(config_prune_ratios) - set(prune_ratios)
            for name in extra_weights:
                print("{} in config file cannot be found".\
                    format(name))


    return configs, prune_ratios



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

    weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy
    weight_ori = copy.copy(weight)

    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    percent = prune_ratio * 100
    if (args.sp_admm_sparsity_type == "irregular") or (args.sp_admm_sparsity_type == "irregular_global"):
        print("irregular pruning...")
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        # weight[under_threshold] = 0
        ww = weight_ori * above_threshold
        return torch.from_numpy(above_threshold), torch.from_numpy(ww)
    else:
        pass

    raise SyntaxError("Unknown sparsity type: {}".format(args.sp_admm_sparsity_type))


def weight_growing(args, name, pruned_weight_np, lower_bound_value, upper_bound_value, update_init_method, mask_fixed_params=None):
    shape = None
    weight1d = None

    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    if upper_bound_value == 0:
        print("==> GROW: {}: to DENSE despite the sparsity type is \n".format(name))
        np_updated_mask = np.ones_like(pruned_weight_np, dtype=np.float32)
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if upper_bound_value == lower_bound_value:
        print("==> GROW: {}: no grow, keep the mask and do finetune \n".format(name))
        non_zeros_updated = pruned_weight_np != 0
        non_zeros_updated = non_zeros_updated.astype(np.float32)
        np_updated_mask = non_zeros_updated
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if (args.sp_admm_sparsity_type == "irregular"):
        # randomly select and set zero weights to non-zero to restore sparsity
        non_zeros_prune = pruned_weight_np != 0

        shape = pruned_weight_np.shape
        weight1d = pruned_weight_np.reshape(1, -1)[0]
        zeros_indices = np.where(weight1d == 0)[0]
        num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(weight1d))
        num_added_zeros = num_added_zeros if num_added_zeros < np.size(zeros_indices) else np.size(zeros_indices)
        num_added_zeros = num_added_zeros if num_added_zeros > 0 else 0
        target_sparsity = 1 - (np.count_nonzero(non_zeros_prune) + num_added_zeros) * 1.0 / np.size(pruned_weight_np)
        indices = np.random.choice(zeros_indices,
                                   num_added_zeros,
                                   replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices),
                                                                                             num_added_zeros,
                                                                                             len(indices)))

        # initialize selected weights
        if update_init_method == "weight":
            pass
        elif update_init_method == "zero":
            # set selected weights to -1 to get corrrect updated masks
            weight1d[indices] = -1
            weight = weight1d.reshape(shape)
            non_zeros_updated = weight != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, target_sparsity))

            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated

            # assign 0 to -1 weight
            weight1d[indices] = 0
            weight = weight1d.reshape(shape)

            # write updated weights back to model
            # self.model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "kaiming":
            assert (False)

        np_updated_mask = np_updated_zero_one_mask
        updated_mask = torch.from_numpy(np_updated_mask).cuda()

        return updated_mask

    else:
        pass

