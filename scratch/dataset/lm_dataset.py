from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # load_dataset 用 Apache Arrow 列式存储，支持内存映射
        # 大数据集不会一次性加载到内存
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        # ① 分词：文本 → token id 列表
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # ② 手动加 BOS/EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # ③ Padding 到固定长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # ④ 构造 labels：PAD 位置设 -100，不参与 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('model')
    train_ds = PretrainDataset('dataset/pretrain_t2t_mini.jsonl', tokenizer, max_length=340)

    loader = DataLoader(
        train_ds,
        batch_size=32,       # 每批 32 个样本
        shuffle=True,        # 随机打乱顺序
        num_workers=8,       # 8 个进程并行加载
        pin_memory=True,     # 锁页内存，加速 CPU→GPU 传输
        drop_last=True       # 丢弃末尾不足一个 batch 的数据
    )

    for input_ids, labels in loader:
        print(f"input_ids: {input_ids.shape}")  # [32, 340]
        print(f"labels:    {labels.shape}")      # [32, 340]
        break