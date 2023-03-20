# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/14 4:03 PM
==================================="""
import gradio as gr
from typing import Tuple
import os
import sys
import torch

import time
import json
from pathlib import Path
from argparse import ArgumentParser
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def generate(prompts, max_gen_len=100, temperature=0.85, top_p=0.95):
    prompts = [prompts]
    results = generator.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    print(results)
    return results


def model_init(args):

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.max_batch_size
    )
    return generator


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/8t/workspace/lchang/models/7B")
    parser.add_argument("--tokenizer_path", type=str, default="/8t/workspace/lchang/models/tokenizer.model")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=1)
    args = parser.parse_args()
    generator = model_init(args)
    examples = [
        ["I believe the meaning of life is"],
        ["Simply put, the theory of relativity states that "],
        ["Building a website can be done in 10 simple steps:\n"],
        ["""Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:"""],
        ["""Translate English to 中文:\nplush girafe => 毛绒长颈鹿\ncheese =>"""]
    ]
    # generate(["请以春天来了为题写一篇作文，题材不限，字数100字。请用中文写。\n春天来了，"], max_gen_len=100, num_return_sequences=1)
    gr.Interface(fn=generate,
                 examples=examples,
                 inputs=["textbox", gr.Slider(10, 512, 150, step=1, label="Max generate length"),
                         gr.Slider(0.1, 1.0, 0.85, label="Temperature"),
                         gr.Slider(0.1, 1.0, 0.95, label="Top P")],
                 outputs="text",
                 title="JoJo Generator",
                 article="Link to <a href='https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api' style='color:blue;' target='_blank\'>参考Prompt</a>",
                 description="请输入一些正确的指令，随心所欲的和我聊天吧").launch(share=True, server_port=8083, server_name="0.0.0.0")


