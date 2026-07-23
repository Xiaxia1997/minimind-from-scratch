import os, sys, time, argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger


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

        if step % args.save_interval == 0 or step == iters:
            model.eval()
            ckp = f'{args.save_dir}/pretrain_{args.hidden_size}.pth'
            torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp)
            model.train()

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
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('../../model')
    Logger(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.amp.autocast('cuda', dtype=dtype))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    for epoch in range(args.epochs):
        train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, args)