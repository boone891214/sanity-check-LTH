import argparse
import os
import shutil
import time
import random
from datetime import datetime

import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

import image_classification.resnet as models
import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *
from image_classification.resnet import add_netarch_parser_arguments
#from image_classification.mobilenetv2 import add_mobilenet_parser_arguments

from prune_utils import *

import dllogger

torch.manual_seed(0)

def add_parser_arguments(parser):
    #model_names = models.resnet_versions.keys()
    #model_configs = models.resnet_configs.keys()

    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali-cpu",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
        + " | ".join(DATA_BACKEND_CHOICES)
        + " (default: dali-cpu)",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        help="model architecture: (default: resnet50), or mobilenetv2",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        #choices=model_configs,
        help="model configs: (default: classic)",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine", "cosine2", "repeat_cosine", "manual"],
        help="Type of LR schedule: {}, {}, {}, {}, {}, {}".format("step", "linear", "cosine", "cosine2", "repeat_cosine", "manual"),
    )

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
        + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        action="store_true",
        help="Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored",
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)
    parser.add_argument("--checkpoint-dir", default=None, type=str, help='dir to save checkpoints, will override self naming')
    parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')


    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )

    parser.add_argument(
        '--id',
        default='',
        type=str,
        help='traing id for different run')
    parser.add_argument(
        '--lr-decay-epochs',
        default='30-60-80',
        type=str,
        help='weight decay at these epochs'
    )
    parser.add_argument(
        "--restart-training",
        action="store_true",
        help="restart training from epoch 0"
    )
    parser.add_argument(
        '--lr-file',
        default=None,
        type=str,
        help='manual learning file'
    )

    parser.add_argument(
        '--imp',
        action='store_true',
        default=False,
        help='enable iterative pruning by hard pruning to higher sparsity ratio at the end of the training'
    )
    parser.add_argument(
        '--imp-ratio',
        type=float,
        default=0.2,
        help='sparsity ratio for every iteration in imp prune'
    )
    parser.add_argument(
        '--imp-round',
        type=int,
        default=None,
        help='define imp round'
    )

def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print(
                "=> loading pretrained weights from '{}'".format(
                    args.pretrained_weights
                )
            )
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    pre_defined_mask = None
    if args.sp_pre_defined_mask_dir is not None:
        if os.path.isfile(args.sp_pre_defined_mask_dir):
            print("\n\n=> loading pre-defined sparse mask from '{}'".format(args.sp_pre_defined_mask_dir))
            pre_defined_mask = torch.load(args.sp_pre_defined_mask_dir, map_location=lambda storage, loc: storage.cuda(args.gpu))
            pre_defined_mask = pre_defined_mask["state_dict"]

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
            )
            if args.restart_training:
                start_epoch = 0
                best_prec1 = 0
                optimizer_state = None
                model_state = checkpoint["state_dict"]
                print("Restart training")

            elif "epoch" in checkpoint and "best_prec1" in checkpoint and "optimizer" in checkpoint:
                start_epoch = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                optimizer_state = checkpoint["optimizer"]
                if (args.sp_admm or args.sp_retrain) and (args.restart_training):
                    start_epoch = 0
                    best_prec1 = 0
                    optimizer_state = None
                model_state = checkpoint["state_dict"]
            else:
                model_state = checkpoint
                start_epoch = 0
                best_prec1 = 0
                optimizer_state = None

            if "epoch" in checkpoint:
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            else:
                print("=> loaded checkpoint '{}' ".format(args.resume))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            time.sleep(10)
            # exit()
            model_state = None
            optimizer_state = None


    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        loss,
        pretrained_weights=pretrained_weights,
        cuda=True,
        fp16=args.fp16,
        memory_format=memory_format,
        args=args,
    )

    for name, W in model_and_loss.model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            print(name, W.shape)

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader

    train_loader, train_loader_len = get_train_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        args.mixup > 0.0,
        start_epoch=start_epoch,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        False,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.raport_file),
                ),
            ],
            start_epoch=start_epoch - 1,
        )

    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    optimizer = get_optimizer(
        list(model_and_loss.model.named_parameters()),
        args.fp16,
        args.lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
    )

    if args.lr_schedule == "step":
        lr_decay_epochs = args.lr_decay_epochs.split('-')
        for i in range(len(lr_decay_epochs)):
            lr_decay_epochs[i] = int(lr_decay_epochs[i])

        lr_policy = lr_step_policy(
            args.lr, lr_decay_epochs, 0.1, args.warmup, logger=logger
        )
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "exponential":
        lr_policy = lr_exponential_policy(args.lr, args.warmup, args.epochs, final_multiplier=0.001, logger=logger)
    elif args.lr_schedule == "manual":
        lr_policy = lr_manual_policy(args.lr_file, args.epochs, logger=logger)

    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level="O1",
            loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
        )

    if args.distributed:
        model_and_loss.distributed()

    if args.evaluate:
        model_state_1 = {}
        keep_prefix = False
        for key, W in model_and_loss.model.named_parameters():
            if key.startswith('module.'):
                keep_prefix = True
        if keep_prefix:
            pass
        else:
            for key in model_state:
                #print(key)
                if key.startswith('module.'):
                    new_key = key[7:]
                    model_state_1[new_key] = model_state[key]
            model_state = copy.copy(model_state_1)


    model_and_loss.load_model_state(model_state)


    if args.sp_retrain:
        if args.retrain_mask_pattern == 'random':
            if args.retrain_mask_sparsity > 0:
                args.id = 'rand-{}-'.format(args.retrain_mask_sparsity) + args.id
            else:
                args.id = 'rand-{}-'.format(args.sp_config_file[9:-5]) + args.id

        elif args.retrain_mask_pattern == 'weight':
            assert not args.resume == None, "Retrain, but no ckpt resumed from, check --resume"


            if args.sp_prune_before_retrain:
                if 'admm' not in args.resume:
                    args.id = '1-shot-{}-{}-LR-{}-'.format(args.sp_admm_sparsity_type,args.sp_config_file[9:-5],args.lr) + args.id

            if 'filter' in args.resume:
                args.id = 'retrain-filter-LR-{}-'.format(args.lr) + args.id
            elif 'irregular' in  args.resume:
                args.id = 'retrain-irregular-LR-{}-'.format(args.lr) + args.id

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir

    else:
        checkpoint_dir = './checkpoints/' + args.arch + '-' + args.widths + '/' + args.id + '/'

    log_dir = './logs/'
    if args.log_filename:
        log_filename = args.log_filename
    else:

        log_filename = log_dir + args.arch + '-' + args.widths + '-' + args.id + '.log'

    print(checkpoint_dir)
    print(log_filename)


    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

        if not args.evaluate and args.save_checkpoints:

            if os.path.isdir(checkpoint_dir) is False:
                os.system('mkdir -p ' + checkpoint_dir)
                print("New folder {} created...".format(checkpoint_dir))

            if os.path.isdir(log_dir) is False:
                os.system('mkdir -p ' + log_dir)
                print("New folder {} created...".format(log_dir))
            #log_filename = log_dir + args.arch + '-' + args.widths + '-' + args.id + '.log'
            log_filename_dir_str = log_filename.split('/')
            log_filename_dir = "/".join(log_filename_dir_str[:-1])
            if not os.path.exists(log_filename_dir):
                os.system('mkdir -p ' + log_filename_dir)
                print("New folder {} created...".format(log_filename_dir))

            with open(log_filename, 'a') as f:
                for arg in sorted(vars(args)):
                    f.write("{}:".format(arg))
                    f.write("{}".format(getattr(args, arg)))
                    f.write("\n")

            print(checkpoint_dir)
            print(log_filename)

    time.sleep(1)

    train_loop(
        model_and_loss,
        optimizer,
        lr_policy,
        train_loader,
        val_loader,
        args.fp16,
        logger,
        should_backup_checkpoint(args),
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=(start_epoch + args.run_epochs)
        if args.run_epochs != -1
        else args.epochs,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        log_file=log_filename,
        args=args,
        pre_defined_mask=pre_defined_mask
    )
    exp_duration = time.time() - exp_start_time
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()
    print("Experiment ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    add_parser_arguments(parser)
    add_netarch_parser_arguments(parser)

    prune_parse_arguments(parser)

    args = parser.parse_args()
    cudnn.benchmark = True

    if len(args.id) == 0:
        now = datetime.now()
        now = str(now)
        now = now.replace(" ","_")
        now = now.replace(":","_")
        now = now.replace("-","_")
        now = now.replace(".","_")
        args.id = now


    print(args)

    main(args)
