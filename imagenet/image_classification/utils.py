import os
import numpy as np
import torch
import shutil
import torch.distributed as dist
import time
import sys

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints #and (epoch < 10 or epoch % 10 == 0)

    return _sbc


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    checkpoint_dir="./checkpoints/",
    backup_filename=None,
    epoch=-1,
    args=None
):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
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



def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    '''
    print(output.data.shape)
    np_data = np.array(output.data.detach().cpu())
    np.set_printoptions(threshold=sys.maxsize)
    print(np_data[0][:20])
    exit()
    #np_pred = np.argmax(np_data,axis=1)


    with open('temp.txt', 'a') as f:
        for _pred in np_pred:
            f.write(str(_pred))
            f.write("\n")
    #input("?")
    '''
    '''
    np_target = np.array(target.cpu())
    #print(np_target)
    #print(np_pred)
    np_res = np.sum(np_target - np_pred == 0)
    prec1 = (np_res + 0.0)*100.0/batch_size
    #print(prec1)
    prec1 = torch.tensor(prec1).cuda()
    '''

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt


def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d
