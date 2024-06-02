import os, sys, time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import Dataset
from torch_utils import seed, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--ckpt_dir", type=str, default=None)

parser.add_argument("--device", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1000000)
parser.add_argument("--save_interval", type=int, default=2000)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--master_addr", type=str, default="localhost")
parser.add_argument("--master_port", type=str, default="29500")

parser.add_argument("--silent", action="store_true", default=False)

parser.add_argument("--no_fix_length", action="store_true", default=False)
parser.add_argument("--no_fix_diameter", action="store_true", default=False)


if __name__ == "__main__":
    settings = parser.parse_args()
    from model import Model

    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device

    seed(settings.seed+settings.rank if settings.rank else settings.seed)
    init_rng_state = get_rng_state(device)
    rng_state = init_rng_state
    if settings.rank is not None:
        os.environ["MASTER_ADDR"] = settings.master_addr
        os.environ["MASTER_PORT"] = settings.master_port
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.set_num_threads(min(torch.get_num_threads()//device_count, 20))

    if settings.train:
        train_dataset = Dataset(settings.train)
        cond_offset = -0.5*(np.max(train_dataset.target, 0) + np.min(train_dataset.target, 0))
        cond_scale = 2/(np.max(train_dataset.target, 0) - np.min(train_dataset.target, 0))
        if settings.workers > 1:
            ind = np.arange(len(train_dataset.data)//settings.workers*settings.workers)
            ind = np.split(ind, settings.workers)[settings.rank]
            print(len(ind), np.min(ind), np.max(ind))
            train_dataset.data = [train_dataset.data[i] for i in ind]
            train_dataset.target = train_dataset.target[ind]
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=settings.batch_size, shuffle=True, drop_last=True
            )
    else:
        train_data = None
        cond_offset = [0, 0, 0]
        cond_scale = [1, 1, 1]

    if settings.test:
        test_dataset = Dataset(settings.test)
        if settings.workers > 1:
            ind = np.arange(len(test_dataset.data)//settings.workers*settings.workers)
            ind = np.split(ind, settings.workers)[settings.rank]
            test_dataset.data = [test_dataset.data[i] for i in ind]
            test_dataset.target = test_dataset.target[ind]
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_size=settings.batch_size, shuffle=False, drop_last=False,
        )
    else:
        test_data = None
    print("ok")

    model = Model(cond_offset, cond_scale,
        fix_l=not settings.no_fix_length,
        fix_r=not settings.no_fix_diameter
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, amsgrad=True)
    print("ok")

    start_epoch = 0
    loss_best = 10000000
    if settings.ckpt_dir:
        ckpt = os.path.join(settings.ckpt_dir, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt_dir, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best)
            if state_dict["loss_eval"] is not None:
                loss_best = state_dict["loss_eval"]
            if train_data is None:
                ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=device)
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            init_rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["init_rng_state"]]
            print("Epoch:", state_dict["epoch"])
            start_epoch = state_dict["epoch"]

    end_epoch = start_epoch+1 if train_data is None or start_epoch >= settings.epochs else settings.epochs


    if settings.rank is not None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=settings.rank, world_size=settings.workers)
        device_ids = None if settings.device is None else [settings.device]
        model = DDP(model, device_ids=device_ids)
        model.state_dict = model.module.state_dict
        model.loss = model.module.loss
        is_chief = settings.rank == 0
    else:
        is_chief = True
    if is_chief and settings.train and settings.ckpt_dir:
        logger = SummaryWriter(log_dir=settings.ckpt_dir)
    else:
        logger = None
    losses = None
    perform_test = test_data is not None

    print("ok")
    amp_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch+1, end_epoch+1):
        if train_data is not None:
            batches = len(train_data.batch_sampler)
            log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
            progress = 20/batches if batches > 20 else 1
            optimizer.zero_grad()
            if not settings.silent: print("Epoch {}/{}".format(epoch, end_epoch))
            tic = time.time()
            set_rng_state(rng_state, device)
            losses = {}
            model.train()
            if not settings.silent:
                sys.stdout.write(log_str.format(
                    cur_batch=0, done="", remain="."*int(batches*progress),
                    time=round(time.time()-tic), comment=""))
            for batch, item in enumerate(train_data):
                item = [_.to(device) for _ in item]
                with autocast():
                    res = model(*item)
                    loss = model.loss(*res)
                amp_scaler.scale(loss["loss"]).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
                for k, v in loss.items():
                    v = v.item()
                    if k not in losses: 
                        losses[k] = v
                    else:
                        losses[k] = (losses[k]*batch+v)/(batch+1)
                if not settings.silent:
                    sys.stdout.write(log_str.format(
                        cur_batch=batch+1, done="="*int((batch+1)*progress),
                        remain="."*(int(batches*progress)-int((batch+1)*progress)),
                        time=round(time.time()-tic),
                        comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()]) + \
                            " - lr: {:e}".format(optimizer.param_groups[0]["lr"])
                    ))
            rng_state = get_rng_state(device)
            if not settings.silent: print()

        if perform_test:
            losses_eval = {}
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                0, len(test_dataset)
            ))
            with torch.no_grad():
                set_rng_state(init_rng_state, device)
                i = 0
                tic = time.time()
                model.eval()
                for batch, item in enumerate(test_data):
                    item = [_.to(device) for _ in item]
                    res = model(*item)
                    loss = model.loss(*res)
                    n = len(item[0])
                    for k, v in loss.items():
                        if k not in losses: 
                            losses_eval[k] = v.item()*n
                        else:
                            losses_eval[k] += v.item()*n
                    i += n
                    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                        i, len(test_dataset)
                    ))
            sys.stdout.write("\r\033[K Loss: {} -- time: {}s".format("/".join(list(map("{:.4f}".format, losses.values()))), int(time.time()-tic)))
            print()


        if is_chief and losses is not None and settings.ckpt_dir:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
            if perform_test:
                for k, v in losses.items():
                    for k, v in losses_eval.items():
                        logger.add_scalar("eval/{}".format(k), v, epoch)
                loss_curr = losses_eval["loss"]
            else:
                loss_curr = losses["loss"]
            best = loss_curr is not None and loss_curr < loss_best
            backup = epoch > 10000 and settings.save_interval and epoch % settings.save_interval == 0
            update = epoch > 500
            if best or backup or update:
                state = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch, loss_eval=loss_best,
                    rng_state=rng_state, init_rng_state=init_rng_state,
                )
            if update:
                torch.save(state, ckpt)
            if backup:
                torch.save(state, ckpt+"-{}".format(epoch))
            if best:
                torch.save(state, ckpt_best)
                loss_best = loss_curr

