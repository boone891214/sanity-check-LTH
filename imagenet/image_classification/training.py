import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as resnet_models
from . import mobilenetv2 as mobilenetv2_models
from . import vgg as vgg_model

from . import utils
import dllogger


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

from prune_utils import *

ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(
        self,
        arch,
        loss,
        pretrained_weights=None,
        cuda=True,
        fp16=False,
        memory_format=torch.contiguous_format,
        args=None,
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))


        if 'resnet' in arch[0]:
            model = resnet_models.build_resnet(arch[0], arch[1], arch[2], args=args)
        elif 'mobilenet' in arch[0]:
            model = mobilenetv2_models.build_mobilenetv2(arch[0], arch[1], arch[2], args=args)
            '''
            width_mult = args.wid_mul
            wid_mul = width_mult.split('-')
            for i in range(len(wid_mul)):
                wid_mul[i] = float(wid_mul[i])
            model = mobilenet_v2.mobilenet_grow(width_mult=wid_mul)
            '''
        elif 'vgg' in arch[0]:
            model = vgg_model.build_vgg(arch[0], arch[1], arch[2], args=args)



        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda().to(memory_format=memory_format)
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)




def get_optimizer(
    parameters,
    fp16,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    bn_weight_decay=False,
):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            verbose=False,
        )

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)

def lr_manual_policy(lr_file, epochs, logger=None):
    assert lr_file != None, "\n\n * Error, learning rate file not specified!"
    assert os.path.isfile(lr_file), "\n\n * Error, learning rate file not found!"
    file = open(lr_file, 'r')
    Lines = file.readlines()
    all_lr = []
    # Strips the newline character
    for line in Lines:
        all_lr.append(float(line))
    if len(all_lr) < epochs:
        fill_up_length = epochs - len(all_lr)
        fill_up = [all_lr[-1]] * fill_up_length
        all_lr.append(fill_up)
    file.close()

    def _lr_fn(iteration, epoch):
        lr = all_lr[epoch]
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(
    model_and_loss, optimizer, fp16, use_amp=False, batch_size_multiplier=1
):
    def _step(input, target, optimizer_step=True, args=None):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        loss = prune_update_loss(loss)

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = (
                optimizer.optimizer
                if isinstance(optimizer, FP16_Optimizer)
                else optimizer
            )
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    param.grad /= batch_size_multiplier
            prune_apply_masks_on_grads()

            optimizer.step()
            optimizer.zero_grad()

            prune_apply_masks()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
    train_loader,
    model_and_loss,
    optimizer,
    lr_scheduler,
    fp16,
    logger,
    epoch,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
    args=None,
):

    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_train_step(
        model_and_loss,
        optimizer,
        fp16,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:

        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)

        prune_update_learning_rate(optimizer, epoch, args)

        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step, args=args)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric("train.loss", to_python_float(loss), bs)
            logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        '''
        np_input = np.array(input_var.detach().cpu())
        print('input-shage:{}'.format(np_input.shape))
        print('input:{}'.format(np_input[0][0][0][:20]))

        np_output = np.array(output.detach().cpu())
        print('output-shage:{}'.format(np_output.shape))
        print('output:{}'.format(np_output[0][:20]))
        exit()
        '''
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True, args=None,
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    top5 = log.AverageMeter()
    val_loss = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)

    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:

        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        top5.record(to_python_float(prec5), bs)
        val_loss.record(to_python_float(loss), bs)

        #print(top1.get_val(), top5.get_val(), val_loss.get_val())
        #time.sleep(2)

        if logger is not None:
            logger.log_metric("val.top1", to_python_float(prec1), bs)
            logger.log_metric("val.top5", to_python_float(prec5), bs)
            logger.log_metric("val.loss", to_python_float(loss), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)

        end = time.time()

    return top1.get_val()[0], top5.get_val()[0], val_loss.get_val()[0], top1.get_val()[1]


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time


def train_loop(
    model_and_loss,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    fp16,
    logger,
    should_backup_checkpoint,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./checkpoints/",
    checkpoint_filename="checkpoint.pth.tar",
    log_file="./logs/log.txt",
    args=None,
    pre_defined_mask=None
):

    prec1 = -1

    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")

    prune_init(args, model_and_loss.model, pre_defined_mask=pre_defined_mask)
    prune_apply_masks() # if wanted to make sure the mask is applied in retrain
    prune_print_sparsity(model_and_loss.model)

    print(start_epoch, end_epoch)

    for epoch in range(start_epoch, end_epoch):
        print("current_lr: ", optimizer.param_groups[0]['lr'])

        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            prune_update(epoch)
            #time.sleep(1)
            train(
                train_loader,
                model_and_loss,
                optimizer,
                lr_scheduler,
                fp16,
                logger,
                epoch,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                args=args,
            )

        if not skip_validation:
            prec1, prec5, val_loss, nimg = validate(
                val_loader,
                model_and_loss,
                fp16,
                logger,
                epoch,
                prof=prof,
                register_metrics=epoch == start_epoch,
                args=args,
            )
            print(prec1, prec5, val_loss, nimg)
        if logger is not None:
            logger.end_epoch()

        if skip_training:
            print("Precision1:{}".format(prec1))
            print("Prec1 from logger metrics: {}".format(logger.metrics["val.top1"]["meter"].get_epoch()))
            continue

        if epoch == end_epoch - 1:
            # ADMM PRUNE
            prune_harden()

        prune_print_sparsity(model_and_loss.model)

        if save_checkpoints and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            log_line = "Epoch:{}, Test_accu:[{},{}] LR:{}\n".format(epoch, \
                        prec1, \
                        logger.metrics["val.top5"]["meter"].get_epoch(), \
                        optimizer.param_groups[0]['lr'])

            with open(log_file, 'a') as f:
                f.write(log_line)
                f.write("\n")

            if not skip_validation:
                is_best = logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
                best_prec1 = max(
                    logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
                )
            else:
                is_best = False
                best_prec1 = 0

            if True: #should_backup_checkpoint(epoch):
                backup_filename = "checkpoint-{}.pth.tar".format(epoch)
            else:
                backup_filename = None
            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": model_and_loss.arch,
                    "state_dict": model_and_loss.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename,
                filename=checkpoint_filename,
                epoch=epoch,
                args=args,
            )

    if args.imp:
        print("\n** Saving for next round IMP \n")
        time.sleep(5)

        best_file_name = os.path.join(args.checkpoint_dir, "model_best.pth.tar")

        assert os.path.isfile(best_file_name), "\n\n * Error, best model not found!"
        print("=> loading checkpoint '{}'".format(best_file_name))

        best_model_state = torch.load(best_file_name, map_location=lambda storage, loc: storage.cuda(args.gpu))
        best_model_state = best_model_state["state_dict"]

        model_and_loss.load_model_state(best_model_state)

        sparsity_ratio = 1 - (1 - args.imp_ratio) ** (args.imp_round + 1)
        sparsity_ratio = np.round(sparsity_ratio, 3)

        args.sp_global_weight_sparsity = sparsity_ratio
        args.sp_admm_update_epoch = 1
        from prune_utils.admm import ADMM
        prune_algo = ADMM(args, model_and_loss.model, logger=None)
        prune_algo.prune_harden()

        utils.save_checkpoint(
            {
                "epoch": epoch,
                "arch": model_and_loss.arch,
                "state_dict": model_and_loss.model.state_dict(),
                "sparsity_ratio": sparsity_ratio,
                "imp_ratio": args.imp_ratio,
                "imp_round": args.imp_round,
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            checkpoint_dir=checkpoint_dir,
            backup_filename="checkpoint_{}_harden_for_next_round_backup.pth.tar".format(args.seed),
            filename="checkpoint_{}_harden_for_next_round.pth.tar".format(args.seed),
            epoch=epoch,
            args=args,
        )




# }}}
