# ---- Chapter 11 补充: Tokenizer 训练 ----
# 写到: scratch/trainer/train_tokenizer.py
#
# 依赖:
#   - pip install tokenizers transformers
#   - 训练数据: dataset/sft_t2t_mini.jsonl（或任何 JSONL 文件）

import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer


DATA_PATH = '../../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../../model_learn_tokenizer/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36   # 特殊 token 总数（含 buffer）


def get_texts(data_path):
    """从 JSONL 文件中逐条提取文本（生成器，不一次性加载全部）"""
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break  # 训练 tokenizer 不需要全量数据，10000 条足够
            try:
                data = json.loads(line)
                # SFT 数据格式：{"conversations": [{"role": ..., "content": ...}, ...]}
                # 取所有 content 拼成一段文本
                contents = [
                    item.get('content')
                    for item in data.get('conversations', [])
                    if item.get('content')
                ]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue


def train_tokenizer(data_path, tokenizer_dir, vocab_size,
                    special_tokens_num=SPECIAL_TOKENS_NUM):
    """训练一个 byte-level BPE tokenizer 并保存

    流程:
    1. 初始化 BPE 模型 + ByteLevel 预分词器
    2. 定义特殊 token（系统级 + 功能性 + buffer）
    3. 在语料上训练 BPE 合并规则
    4. 保存 tokenizer.json + tokenizer_config.json
    """
    # ---- 1. 初始化 BPE 模型 + 字节级预分词 ----
    tokenizer = Tokenizer(models.BPE())
    # ByteLevel 预分词：把文本按空格/标点切成粗粒度的"词"
    # add_prefix_space=False: 不在每个词前加空格前缀
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ---- 2. 定义特殊 token ----
    # 第一类：系统级（21 个）
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
        "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    # 第二类：功能性（6 个）
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    # 第三类：预留 buffer
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens

    # ---- 3. 创建 BPE 训练器 ----
    trainer_obj = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        # initial_alphabet: 确保 256 个字节全在初始词表中
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # 特殊 token 占用词表前 N 个位置（id=0,1,2,...）
        special_tokens=all_special_tokens
    )

    # ---- 4. 训练 ----
    texts = get_texts(data_path)
    # train_from_iterator: 从生成器迭代读取文本，流式训练
    tokenizer.train_from_iterator(texts, trainer=trainer_obj)
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    # 把系统级 token 标记为 special（不会被分词器拆分）
    tokenizer.add_special_tokens(special_tokens_list)

    # ---- 5. 保存 tokenizer.json ----
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # ---- 6. 修正 added_tokens 的 special 标记 ----
    # additional_tokens（如 <tool_call>）不是 special token
    # 需要把它们的 special 字段设为 false
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # ---- 7. 生成 tokenizer_config.json ----
    added_tokens_decoder = {}
    for token in all_special_tokens:
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False, "normalized": False,
            "rstrip": False, "single_word": False,
            "special": token in special_tokens_list
        }

    config = {
        "add_bos_token": False,        # 我们手动加，不让 tokenizer 自动加
        "add_eos_token": False,        # 同上
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [
            t for t in special_tokens_list if t != "<|endoftext|>"
        ],
        "bos_token": "<|im_start|>",   # BOS = 对话轮次开始
        "eos_token": "<|im_end|>",     # EOS = 对话轮次结束
        "pad_token": "<|endoftext|>",  # PAD = 填充
        "unk_token": "<|endoftext|>",  # UNK 也用 PAD（byte-level 不会有 UNK）
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": 131072,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        # chat_template 省略（完整版见上游 train_tokenizer.py）
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"Tokenizer training completed. Saved to {tokenizer_dir}")


def eval_tokenizer(tokenizer_dir):
    """验证训好的 tokenizer：encode/decode 一致性 + 压缩率"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # ---- 1. ChatML 格式验证 ----
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自月球"},
        {"role": "user", "content": "你到底来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print('-' * 80)
    print(prompt)
    print('-' * 80)

    # ---- 2. encode → decode 一致性 ----
    print(f"词表大小: {len(tokenizer)}")
    encoded = tokenizer(prompt)
    print(f"encode 长度: {len(encoded['input_ids'])}")
    decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)
    print(f"decode 一致: {decoded == prompt}")

    # ---- 3. 压缩率测试（字符数 / Token数）----
    print('-' * 80)
    print("压缩率测试（Chars/Tokens）:")
    test_texts = [
        # 中文
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        # 英文
        "Large language models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence.",
        # 混合
        "Python 是一种高级编程语言，widely used in data science and machine learning。",
    ]
    total_ratio = 0
    for i, text in enumerate(test_texts):
        tokens = tokenizer.encode(text)
        ratio = len(text) / len(tokens)
        total_ratio += ratio
        print(f"  样本{i+1} | 字符:{len(text):3} | Tokens:{len(tokens):3} | 压缩率:{ratio:.2f}")
    print(f"  平均压缩率: {total_ratio / len(test_texts):.2f}")

    # ---- 4. 流式解码测试（字节缓冲）----
    print('-' * 80)
    print("流式解码测试:")
    input_ids = encoded['input_ids'][:20]  # 取前 20 个 token
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        # � 是 Unicode 替换字符，表示字节还不完整
        if current_decode and '�' not in current_decode:
            print(f"  Token ID: {token_cache} -> {current_decode}")
            token_cache = []


if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
