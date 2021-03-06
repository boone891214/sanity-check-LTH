import time
import random
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

from models.resnet_grasp import resnet32
from models.resnet20_cifar import resnet20
from models.vgg_grasp import vgg16, vgg19
from models.mobilenet import mobilenetv1

from prune_utils import *
from prune_utils.prune_main import prune_harden

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='[cifar10， cifar100]')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.001,
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.3, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--rho', type=float, default = 0.0001,
                    help ="Just for initialization")
parser.add_argument('--pretrain-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pretrain')
parser.add_argument('--pruning-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pruning')
parser.add_argument('--remark', type=str, default=None,
                    help='optimizer used (default: adam)')
parser.add_argument('--save-model', type=str, default=None,
                    help='optimizer used (default: adam)')

parser.add_argument('--patternNum', type=int, default=8, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--rand-seed', action='store_true', default=False,
                    help='use random seed')
parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')
parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument('--imp', action='store_true', default=False,
                    help='enable iterative pruning by hard pruning to higher sparsity ratio at the end of the training')
parser.add_argument('--imp-ratio', type=float, default=0.2,
                    help='sparsity ratio for every iteration in imp prune')
parser.add_argument('--imp-round', type=int, default=None,
                    help='define imp round')


parser.add_argument("--evaluate", action="store_true", help="evaluate checkpoint/model")
parser.add_argument('--restart-training', action='store_true', default=False,
                    help='reset optimizer state')



prune_parse_arguments(parser)
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.rand_seed:
    seed = random.randint(1, 999)
    print("Using random seed:", seed)
else:
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    print("Using manual seed:", seed)

if not os.path.exists("./model_init/"):
    os.makedirs("./model_init/")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



train_dataset = None
test_dataset = None

if args.dataset == "cifar10":
    train_dataset = datasets.CIFAR10('./data.cifar10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Pad(4),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ]))
    test_dataset = datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ]))

elif args.dataset == "cifar100":
    train_dataset = datasets.CIFAR100('./data.cifar100', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                            ]))
    test_dataset = datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                            ]))



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)



def train(model, train_loader, criterion, scheduler, optimizer, epoch, maskretrain, masks):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        scheduler.step()
        prune_update_learning_rate(optimizer, epoch, args)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        output = model(input)

        ce_loss = criterion(output, target, smooth=args.smooth)

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        ce_loss.backward()

        # dynamic sparse training (DST): update mask at certain point
        prune_update(epoch, batch_idx=i)

        # make sure the corresponding gradient of the pruned weights are zero
        prune_apply_masks_on_grads()

        optimizer.step()

        optimizer.zero_grad()

        # make sure the pruned weights are zero after weight update
        prune_apply_masks()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss = criterion(output, target)
            # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    prec1 = float(100. * correct) / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), prec1))



    return (float(100 * correct) / float(len(test_loader.dataset)))

def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    checkpoint_dir="./checkpoints/",
    backup_filename=None,
    epoch=-1,
    args=None
):

    filename = os.path.join(checkpoint_dir, filename)
    print("SAVING {}".format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint_dir, "model_best.pth.tar")
        )
        shutil.copyfile(
            filename, os.path.join(checkpoint_dir, "best_{}.pth.tar".format(epoch))
        )
    if backup_filename is not None:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))



class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def main():
    all_acc = [0.000]
    epoch_cnt = 0

    if args.cuda:
        if args.arch == "resnet":
            if args.depth == 20:
                model = resnet20(dataset=args.dataset)
            elif args.depth == 32:
                model = resnet32(dataset=args.dataset)
            else:
                sys.exit("resnet doesn't implement those depth!")
        elif args.arch == "vgg":
            if args.depth == 16:
                model = vgg16(dataset=args.dataset)
            elif args.depth == 19:
                model = vgg19(dataset=args.dataset)
            else:
                sys.exit("vgg doesn't have those depth!")
        elif args.arch == "mobilenetv1":
            model = mobilenetv1(dataset=args.dataset)

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
        model.cuda()

    model_state = None
    optimizer_state = None
    scheduler_state = None
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if "state_dict" in checkpoint or "optimizer" in checkpoint or "scheduler" in checkpoint:
                if not args.restart_training:
                    print("=> loading checkpoint with restarting info")
                    start_epoch = checkpoint["epoch"]
                    model_state = checkpoint['state_dict']
                    optimizer_state = checkpoint['optimizer']
                    scheduler_state = checkpoint['scheduler']
                else:
                    start_epoch = 0
                    model_state = checkpoint['state_dict']
            else:
                model_state = checkpoint
                start_epoch = 0

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            time.sleep(5)
            model_state = None

    if not model_state is None:
        model.load_state_dict(model_state)
        print("=> loaded model_state")

    pre_defined_mask = None
    if args.sp_pre_defined_mask_dir is not None:
        if os.path.isfile(args.sp_pre_defined_mask_dir):
            print("\n\n=> loading pre-defined sparse mask from '{}'".format(args.sp_pre_defined_mask_dir))
            pre_defined_mask = torch.load(args.sp_pre_defined_mask_dir)
            if "state_dict" in pre_defined_mask.keys():
                pre_defined_mask = pre_defined_mask["state_dict"]
            else:
                pre_defined_mask = pre_defined_mask

    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    # args.smooth = args.smooth_eps > 0.0
    # args.mixup = config.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    optimizer = None
    if (args.optmzr == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif (args.optmzr == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                         eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [80, 120]
        # epoch_milestones = [30, 60, 90, 120]

        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[i * len(train_loader) for i in epoch_milestones],
                                                   gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr,
                                           total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)

    if not optimizer_state is None:
        optimizer.load_state_dict(optimizer_state)
        print("=> loaded optimizer_state")
    if not scheduler_state is None:
        scheduler.load_state_dict(scheduler_state)
        print("=> loaded scheduler_state")

    log_filename = args.log_filename
    checkpoint_dir = args.save_model
    print(checkpoint_dir)
    print(log_filename)

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

    if os.path.isdir(checkpoint_dir) is False:
        os.system('mkdir -p ' + checkpoint_dir)
        print("New folder {} created...".format(checkpoint_dir))

    print("\n>_ Saving initial model ...\n")
    torch.save(model.state_dict(),
               "./model_init/{}_{}{}_{}_lr{}_{}_seed{}_{}.pt".format(args.dataset, args.arch, args.depth, args.optmzr, args.lr, args.lr_scheduler, seed, args.remark))


    # initialize prune function, one-shot prune the model before training
    prune_init(args, model, pre_defined_mask=pre_defined_mask)
    prune_apply_masks()
    prune_print_sparsity(model)

    best_file_name = []
    for epoch in range(start_epoch, args.epochs):
        # prune_update(epoch)
        if not args.evaluate:
            train(model, train_loader, criterion, scheduler, optimizer, epoch, maskretrain=False, masks={})
        prec1 = test(model)
        epoch_cnt += 1
        prune_print_sparsity(model)

        if epoch == args.epochs - 1:
            # ADMM PRUNE
            prune_harden(args, model)

        if log_filename is not None:
            log_line = "Epoch:{}, Test_accu:[{}] LR:{}\n".format(epoch,
                                                                 prec1,
                                                                 optimizer.param_groups[0]['lr'])

            with open(log_filename, 'a') as f:
                f.write(log_line)
                f.write("\n")

        if False:  # should_backup_checkpoint(epoch):
            backup_filename = "checkpoint-{}.pth.tar".format(epoch)
        else:
            backup_filename = None

        if checkpoint_dir is not None:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "prec1": prec1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                },
                is_best=False,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename,
                filename="checkpoint_{}.pth.tar".format(seed),
                epoch=epoch,
                args=args,
            )

            if prec1 > max(all_acc):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
                filename = "./{}cifar10_{}{}_acc_{:.3f}_{}_lr{}_{}_epoch{}_seed{}_{}.pt".format(args.save_model, args.arch,
                                                                                                args.depth, prec1,
                                                                                                args.optmzr, args.lr,
                                                                                                args.lr_scheduler,
                                                                                                epoch_cnt, seed,
                                                                                                args.remark)
                torch.save(model.state_dict(), filename)
                best_file_name.append(filename)
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_acc)))
                if len(all_acc) > 1:
                    os.remove(
                        "./{}cifar10_{}{}_acc_{:.3f}_{}_lr{}_{}_epoch{}_seed{}_{}.pt".format(args.save_model, args.arch,
                                                                                             args.depth, max(all_acc),
                                                                                             args.optmzr, args.lr,
                                                                                             args.lr_scheduler, best_epoch,
                                                                                             seed, args.remark))
                best_epoch = epoch_cnt
        all_acc.append(prec1)

    '''
        code for IMP pruning. If true, prune the model to higher level AT THE END of the current round and save 
        a 'higher-sparsity'model, which will serve as the mask-model for the NEXT round.
    '''
    if args.imp:
        print("\n** Saving for next round IMP \n")

        if os.path.isfile(best_file_name[-1]):
            print("=> loading checkpoint '{}'".format(best_file_name[-1]))
            best_model_state = torch.load(best_file_name[-1])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

        model.load_state_dict(best_model_state)

        sparsity_ratio = 1 - (1 - args.imp_ratio) ** (args.imp_round + 1)
        sparsity_ratio = np.round(sparsity_ratio, 3)

        args.sp_global_weight_sparsity = sparsity_ratio
        args.sp_admm_update_epoch = 1
        prune_harden(args, model)

        save_checkpoint(
            {
                "epoch": epoch,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "sparsity_ratio": sparsity_ratio,
                "imp_ratio": args.imp_ratio,
                "imp_round": args.imp_round
            },
            is_best=False,
            checkpoint_dir=checkpoint_dir,
            backup_filename=backup_filename,
            filename="checkpoint_{}_harden_for_next_round.pth.tar".format(seed),
            epoch=epoch,
            args=args,
        )


if __name__ == '__main__':
    main()