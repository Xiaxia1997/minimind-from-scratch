# ---- Chapter 12b: 知识蒸馏训练脚本 ----
# 写到: scratch/trainer/train_distillation.py
#
# 依赖:
#   - scratch/model/model_minimind.py (Chapter 08 完整模型)
#   - scratch/dataset/lm_dataset.py   (Chapter 12 SFTDataset)
#   - scratch/trainer/trainer_utils.py (Chapter 09 get_lr / Logger)

import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger


# ============================================================
# 蒸馏 Loss
# ============================================================

def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """KL 散度蒸馏 loss，student 拟合 teacher 的 soft label

    Args:
        student_logits: [N, V] student 的原始 logits（N=有效token数, V=vocab_size）
        teacher_logits: [N, V] teacher 的原始 logits
        temperature: 蒸馏温度，越高分布越平滑

    Returns:
        标量 loss = T² × KL(student_T || teacher_T)
    """
    # teacher 的 soft label：高温 softmax → 更平滑的概率分布
    # .detach() 阻止梯度流入 teacher（teacher 不更新）
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # student 的 log 概率：和 teacher 用相同温度
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度：P * (log P - log Q)
    # F.kl_div 接收 (log Q, P)，即 (student_log_probs, teacher_probs)
    # reduction='batchmean'：先求和再除以 batch size
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # T² 补偿：高温 softmax 使梯度缩小 1/T，乘 T² 补回来
    return (temperature ** 2) * kl


# ============================================================
# 训练一个 epoch
# ============================================================

def train_epoch(model, teacher_model, loader, optimizer, scaler, autocast_ctx,
                epoch, lm_config_student, args):
    model.train()
    # teacher 全程冻结：eval 模式 + 关闭梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    iters = len(loader)
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)  # [B, S]
        labels = labels.to(args.device)         # [B, S]

        # loss_mask: 标记哪些位置是有效 token（非 -100）
        # labels[..., 1:] 是移位后的 label，和 logits[..., :-1] 对齐
        loss_mask = (labels[..., 1:] != -100).float()  # [B, S-1]

        # LR 调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- student 前向 ----
        with autocast_ctx:
            res = model(input_ids)
            # 移位：logits[..., :-1] 对应 labels[..., 1:]
            student_logits = res.logits[..., :-1, :].contiguous()  # [B, S-1, V]

        # ---- teacher 前向（不存计算图，省显存）----
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # vocab_size 对齐：teacher 可能有更大的词表
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ---- 计算 Loss ----

        # 1) CE Loss：student vs ground truth
        shift_labels = labels[..., 1:].contiguous()  # [B, S-1]
        loss_mask_flat = loss_mask.view(-1)           # [B*(S-1)]
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # [B*(S-1), V]
            shift_labels.view(-1),                              # [B*(S-1)]
            ignore_index=-100,
            reduction='none'                                    # 逐 token loss
        )
        # 手动 mask：只对有效 token 求平均
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # MoE 辅助 loss
        if lm_config_student.use_moe:
            ce_loss_total = ce_loss_raw + res.aux_loss
        else:
            ce_loss_total = ce_loss_raw

        # 2) 蒸馏 Loss：student vs teacher
        if teacher_model is not None:
            # 只在有效 token 上算 KL
            # loss_mask_flat == 1 的位置是有效 token
            distill_loss_val = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=args.temperature
            )
        else:
            distill_loss_val = torch.tensor(0.0, device=args.device)

        # 3) 总 Loss = α * CE + (1-α) * KL
        loss = (args.alpha * ce_loss_total + (1 - args.alpha) * distill_loss_val)
        loss = loss / args.accumulation_steps

        # ---- 反向 + 更新 ----
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if step % args.log_interval == 0 or step == iters:
            real_loss = loss.item() * args.accumulation_steps
            eta_min = (time.time() - start_time) / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                   f'loss:{real_loss:.4f} ce:{ce_loss_raw.item():.4f} '
                   f'distill:{distill_loss_val.item():.4f} lr:{lr:.8f} '
                   f'eta:{eta_min:.1f}min')

        # ---- 保存 ----
        if step % args.save_interval == 0 or step == iters:
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.hidden_size}{moe_suffix}.pth'
            torch.save(
                {k: v.half().cpu() for k, v in model.state_dict().items()},
                ckp
            )
            model.train()

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss_val, loss

    # epoch 末尾：处理残余累积梯度
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", default="../../out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--data_path", default="../../dataset/sft_t2t_mini.jsonl")
    # student 配置
    parser.add_argument("--student_hidden_size", type=int, default=768)
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--student_use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--from_student_weight", default="full_sft")
    # teacher 配置
    parser.add_argument("--teacher_hidden_size", type=int, default=768)
    parser.add_argument("--teacher_num_layers", type=int, default=8)
    parser.add_argument("--teacher_use_moe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--from_teacher_weight", default="full_sft")
    # 蒸馏超参
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CE 权重，总 loss = alpha*CE + (1-alpha)*KL")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="蒸馏温度（推荐 1.0-2.0）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 1. student 模型 ----
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.student_use_moe),
    )
    model = MiniMindForCausalLM(lm_config_student).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('../../model')
    Logger(f'Student params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M')

    # ---- 2. teacher 模型（冻结）----
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.teacher_use_moe),
    )
    teacher_model = MiniMindForCausalLM(lm_config_teacher).to(args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'Teacher params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f}M')

    # ---- 3. 数据 + 优化器 ----
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---- 4. 混合精度 ----
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.amp.autocast('cuda', dtype=dtype))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # ---- 5. 训练 ----
    for epoch in range(args.epochs):
        train_epoch(model, teacher_model, loader, optimizer, scaler, autocast_ctx,
                    epoch, lm_config_student, args)
