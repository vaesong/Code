import os
import random
import argparse
import torch
import json
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import itertools
from filelock import FileLock
from typing import List, Tuple, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
# from robotic_transformer_pytorch import MaxViT, RT1
from my_network import MaxViT, RT1

from Dataset import RLBench_RT1
from tqdm import tqdm, trange
from utils import load_episodes,load_instructions
from Utils import (Loss_Metrics, 
    get_log_dir, 
    CheckpointCallback, 
    count_parameters, 
    tokenize_act_values,
    Actioner,
    RLBenchEnv_RT1)
torch.set_num_threads(4)

def main(gpu, ngpus_per_node, args):
    
    # 首先处理分布式的问题
    # 这里的 gpu 如果是分布式，就是第几个进程（0，1，2，...），如果不是分布式，就是指定的 gpu
    args.gpu = args.gpu_list[gpu]  if args.distributed  else gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # 不同的进程的model参数初始化要相同，可以用同样的随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 处理 checkpoint 存放的路径,只有主进程才生成 tensorboard 文件
    if args.rank == 0:
        log_dir = get_log_dir(args.output / args.mode)
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    device = torch.device(args.gpu)
    
    with open(Path(__file__).parent / "episodes2.json") as fid:
        eps_dict = json.load(fid)
        max_eps_dict = eps_dict["max_episode_length"]

    vit = MaxViT(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (2, 2, 2),
        window_size = 8,
        channels = 4,
    )

    model = RT1(
        vit = vit,
        num_actions = 8,
        depth = 3,
        heads = 4,
        dim_head = 64,
        cond_drop_prob = 0.2,
        cnn_depth=4,
        max_eps_dict=max_eps_dict,
    )

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)        
        
    optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay=args.wt_decay)
    # {"params": list(self.z_dict.values()), "weight_decay": 5e-4}
    model.train()

    # 用来保存模型
    model_dict = {
        "weight": model.module.state_dict() if args.distributed else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    if args.rank == 0:
        checkpointer = CheckpointCallback(log_dir, model_dict)
    else:
        checkpointer = None

    # 加载模型
    if args.checkpoint is not None:
        print("Load from checkpoint ........")
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        if args.distributed:
            model.module.load_state_dict(model_dict["weight"])
        else:
            model.load_state_dict(model_dict["weight"])
        optimizer.load_state_dict(model_dict["optimizer"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/args.warmup_steps, 1))
    
    # 设置损失函数
    loss_and_metrics = Loss_Metrics(args)

    # print(model)
    print("Number of parameters:")
    model_params = count_parameters(model)
    print("- model", model_params)
    print("Total", model_params)

    taskvar = list(itertools.product(args.tasks, args.variations))
        
    with open(args.instructions) as fid:
        instruction = json.load(fid)

    train_dataset = RLBench_RT1(
            root=args.data_dir,
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=args.maxAction,
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(  
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),
                num_workers=args.num_workers, 
                pin_memory=True, 
                sampler=train_sampler, 
                drop_last=True,
                persistent_workers=True
                ) 

    iter_loader = iter(train_loader)
    device = next(model.parameters()).device
    with trange(args.epochs, ncols=170) as tbar:
        for step_id in tbar:              
            try:
                sample = next(iter_loader)
            except StopIteration:
                train_sampler.set_epoch(step_id)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)
                
            rgbs = sample["rgbs"].to(device)    # torch.Size([B, T, C, H, W]) -> B, T, N, C, H, W
            instructions = sample["instr"]   # (B, str)
            pc_obs = sample["pcds"].to(device)
            padding_mask = sample["padding_mask"].to(device)
            frame_id = sample["frame_id"]
            tasks = sample["task"]
            # one_hot = sample["one_hot"].to(device)
            z_offset = torch.stack([model.module.z_dict[task][fid] for task, fid in zip(tasks, frame_id)]).to(device)

            # action_preds = model(rgbs, instructions, pc_obs, padding_mask) # (2, 6, 8, 256) # (batch, frames, actions, bins)
            action_preds = model(rgbs, instructions, pc_obs, padding_mask, z_offset)
            # action_preds = model(rgbs, instructions, pc_obs, padding_mask, one_hot)

            train_losses = loss_and_metrics.compute_loss(action_preds["action"], sample)
            metrics = loss_and_metrics.compute_metrics(action_preds["action"], sample)
            train_losses["total"] = sum([train_losses["position"], train_losses["rotation"], train_losses["gripper"]])
            
            optimizer.zero_grad()
            train_losses["total"].backward()
                        
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # total_norm = 0
            # norm_type = float(2) 
            # for p in model.parameters():
            #     param_norm = p.grad.data.norm(norm_type)
            #     total_norm += param_norm.item() ** norm_type
            # total_norm = total_norm ** (1. / norm_type)
            # print(total_norm, "=====================================")
            # total_norm = 0
            # for param in model.parameters():
            #     total_norm += param.grad.item()
            # print(total_norm, "=====================================")
            
            optimizer.step()
            scheduler.step()


            if writer is not None:
                for n, l in train_losses.items():
                    writer.add_scalar(f"train-loss/{n}", l, step_id)

                for n, l in metrics.items():
                    writer.add_scalar(f"train-metrics/{n}", l, step_id)


            if checkpointer is not None and (step_id + 1) % args.checkpoint_period == 0:
                checkpointer(step_id + 1)

            # tbar.set_postfix(l=float(action_loss))
            tbar.set_postfix(total_loss=float(train_losses["total"]), 
                            position_loss=float(train_losses["position"]), 
                            position_metric=float(metrics["position"]))


def test(args):

    device = torch.device(args.gpu)
    # 不同的进程的model参数初始化要相同，可以用同样的随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(Path(__file__).parent / "episodes.json") as fid:
        eps_dict = json.load(fid)
        max_eps_dict = eps_dict["max_episode_length"]
        
    vit = MaxViT(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (2, 2, 2),  #(2, 5, 2)
        window_size = 8,
        channels = 4,
    )

    model = RT1(
        vit = vit,
        num_actions = 8,
        depth = 3,  # 5
        heads = 4,  # 8
        dim_head = 64,
        cond_drop_prob = 0.2,
        cnn_depth = 4,
        max_eps_dict = max_eps_dict,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay=args.wt_decay)

    # 加载模型
    if args.checkpoint is not None:
        print("Load from checkpoint ........")
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in model_dict["weight"].items():
        #     name = k[7:]    # remove module
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(model_dict["optimizer"])

    env = RLBenchEnv_RT1(
        data_path="/home/liuchang/projects/tianyu/hiveformer/raw/dataset/0",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("wrist", "left_shoulder", "right_shoulder"),
        image_size=(128,128),
        headless=True,
        gripper_pose="attn",
    )
    
    
    if args.save_attn:
        log_dir = get_log_dir(args.output/args.mode)
        log_dir.mkdir(exist_ok=True, parents=True)
    else:
        log_dir = None

    with open(args.instructions) as fid:
        instruction = json.load(fid)

    actioner = Actioner(
        model={"model": model},  # type: ignore
        instructions=instruction,
    )

    # 添加存储成功率的文件
    output_file_name = "/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/test.txt"
    file = open(output_file_name, "a")            

    for task_str in args.tasks:
        for variation_id in args.variations:
            
            if args.use_demo:
                episodes_dir = Path('/home/liuchang/projects/tianyu/hiveformer/raw/dataset/0') / task_str / \
                    f"variation{variation_id}" / "episodes"
                demos = []
                for ep in tqdm(episodes_dir.glob('episode*')):
                    episode_id = int(ep.stem[7:])
                    demo = env.get_demo(task_str, variation_id, episode_id)
                    demos.append(demo[0])
                    # if len(demos) > 1:
                    #     break
                    num_demos = len(demos)
            else:
                demos = None
                num_demos = 100
        
            success_rate = env.evaluate(
                task_str,
                actioner=actioner,
                max_episodes=max_eps_dict.get(task_str, 6),
                variation=variation_id,
                num_demos=num_demos,
                demos=demos,
                log_dir=log_dir,
                save_attn=args.save_attn,
                save_video=args.save_video,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
            file.write(f"{task_str}-{variation_id}, na, seed={args.seed}, success_rate={success_rate}\n")
            file.flush()

    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.set_num_threads(1)
    # dataset 超参
    parser.add_argument('--data_dir', type=str, default='/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/dataset/packaged', help='dataset root path')
    parser.add_argument('--instructions', type=str, default='/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/instructions.json', help='instruction path')
    parser.add_argument('--tasks', type=list, default=[
        # "reach_target",
        # "push_button",
        # "pick_and_lift",
        # "pick_up_cup",
        # "put_knife_on_chopping_board",
        # "take_money_out_safe",
        "put_money_in_safe",
        # "take_umbrella_out_of_umbrella_stand",
        # "stack_wine",
        # "slide_block_to_target",
        ], help='tasks')

    parser.add_argument('--variations', type=Tuple[int, ...], default=(0,))
    parser.add_argument('--output', type=Path, default='/home/liuchang/projects/VLMbench/VLMbench/hiveformer-raw/rt1/xp', help='output path')
    parser.add_argument('--checkpoint', type=Path, default=None, help='checkpoint path')
    
    parser.add_argument('--checkpoint_period', type=int, default=10_000, help='checkpoint_period')


    parser.add_argument('--img_size',nargs='+', type=int, default=[360,360])

    # 训练的超参
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100_000)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    #模型的超参
    parser.add_argument('--num_classes', type=int, default=1000, help='Vit mlp output')
    parser.add_argument('--dim_conv_stem', type=int, default=64, help='conv2D output channels')
    parser.add_argument('--dim', type=int, default=96, help='vit hidden dim')
    parser.add_argument('--dim_head', type=int, default=32, help='dimension head')
    parser.add_argument('--depth', type=Tuple[int, ...], default=(2, 5, 2), help='depth of Vit, sum 20 layers')
    parser.add_argument('--window_size', type=int, default=8, help='window size of Vit')


    parser.add_argument('--maxAction', type=int, default=6)
    parser.add_argument('--act_dim', type=int, default=8)

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--save_attn', type=bool, default=False)
    parser.add_argument('--use_demo', type=bool, default=False)
    parser.add_argument('--save_video', type=bool, default=False)

    # 分布式
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23464')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--gpu_list', type=list, default=[4,5,6,7], help='tasks')
    parser.add_argument('--gpu_number', type=int, default=0)
    parser.add_argument('--ngpus_per_node', type=int, default=0)

    args = parser.parse_args()

    #如果没有设置数量，就自动检测，得到总共该节点有几块 gpu 可用
    ngpus_per_node = torch.cuda.device_count() if len(args.gpu_list)==0 else len(args.gpu_list)
    args.ngpus_per_node = ngpus_per_node

    if args.mode == "train":
        if args.distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            main(args.gpu, ngpus_per_node, args)
    else:
        test(args)