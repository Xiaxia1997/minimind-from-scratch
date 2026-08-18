import os, sys, time, argparse
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger, init_distributed_mode, setup_seed, is_main_process


def train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, args):
    model.train()
    iters = len(loader)
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # Step 8: LR 调度
        current_lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Step 5: 混合精度前向
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps      # Step 6: 梯度累积

        scaler.scale(loss).backward()                  # Step 5: scaler 包住 backward

        # Step 6: 每 N 步才更新
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                 # Step 5: 恢复真实梯度
            torch.nn.utils.clip_grad_norm_(            # Step 7: 梯度裁剪
                model.parameters(), args.grad_clip)
            scaler.step(optimizer)                     # Step 3: AdamW 更新
            scaler.update()
            optimizer.zero_grad(set_to_none=True)      # Step 2: 清零

        if step % args.log_interval == 0 or step == iters:
            real_loss = loss.item() * args.accumulation_steps
            eta_min = (time.time() - start_time) / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                   f'loss:{real_loss:.4f} lr:{current_lr:.8f} eta:{eta_min:.1f}min')

        # DDP: 只让 rank 0 保存权重，避免多卡重复写同一个文件
        if step % args.save_interval == 0 or step == iters:
            if is_main_process():
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model.eval()
                ckp = f'{args.save_dir}/pretrain_{args.hidden_size}.pth'
                torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
                raw_model.train()

        del input_ids, labels, res, loss

    # epoch 末尾：处理残余累积梯度
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="../../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--data_path", default="../../dataset/pretrain_t2t_mini.jsonl")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- DDP Step 1: 初始化分布式环境 ----
    # torchrun 自动设 RANK/LOCAL_RANK/WORLD_SIZE 环境变量
    # 非 DDP 模式下 init_distributed_mode() 直接返回 0，后续代码走单卡路径
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 每张卡用不同的种子，保证各卡看到不同的数据增强（如果有的话）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('../../model')
    Logger(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # ---- DDP Step 2: DistributedSampler 切分数据 ----
    # DDP 模式下用 DistributedSampler 把数据集按 rank 均匀切分
    # 单卡模式下 sampler=None，由 DataLoader 的 shuffle=True 负责打乱
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    num_workers = 0 if args.device == "mps" else 4
    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=(train_sampler is None),
                        sampler=train_sampler,
                        num_workers=num_workers, pin_memory=(args.device != "mps"),
                        drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if "cuda" in args.device:
        autocast_ctx = torch.amp.autocast('cuda', dtype=dtype)
        scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    elif args.device == "mps":
        autocast_ctx = nullcontext()
        scaler = torch.amp.GradScaler(enabled=False)
    else:
        autocast_ctx = nullcontext()
        scaler = torch.amp.GradScaler(enabled=False)

    # ---- DDP Step 3: DistributedDataParallel 包装模型 ----
    # DDP 在每张卡上持有完整的模型副本，反向传播时自动 AllReduce 梯度
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(args.epochs):
        # ---- DDP Step 4: 每 epoch 更新 sampler 的种子 ----
        # 不调 set_epoch 的话，每个 epoch 各卡拿到的数据切分方式相同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, args)

    # ---- DDP Step 5: 清理分布式进程 ----
    # barrier 等所有卡都训完，再一起销毁进程组
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
