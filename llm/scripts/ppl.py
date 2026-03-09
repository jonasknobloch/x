#!/usr/bin/env python3

"""
Adapted from https://github.com/huggingface/transformers/blob/0a0ac7a2875cf481f1edf77552a7c5a6ae1399a5/docs/source/en/perplexity.md
"""

import argparse

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='openai-community/gpt2-large')
    parser.add_argument('--dataset', default='wikitext:wikitext-2-raw-v1:test')
    parser.add_argument('--max-tokens', type=int, default=0)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--match-hf', action='store_true')

    args = parser.parse_args()

    parts = args.dataset.split(':')

    if len(parts) > 3:
        parser.error("invalid --dataset value; expected 'path[:name[:split]]'")

    args.dataset_path = parts[0]
    args.dataset_name = parts[1] if len(parts) > 1 and parts[1] else None
    args.dataset_split = parts[2] if len(parts) > 2 and parts[2] else 'test'

    return args


def main():
    args = parse_args()

    device = Accelerator().device

    model_id = args.model
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    data = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
    )

    encodings = tokenizer('\n\n'.join(data['text']), return_tensors='pt')

    eval_max_tokens = encodings.input_ids.shape[1] if args.max_tokens == 0 else args.max_tokens
    eval_input_ids = encodings.input_ids[:, :eval_max_tokens]

    max_length = model.config.n_positions
    stride = args.stride
    seq_len = eval_input_ids.size(1)

    nll_sum = torch.tensor(0.0, device=device)
    n_tokens = 0
    prev_end_loc = 0

    match_hf = args.match_hf

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

        input_ids = eval_input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift

        if not match_hf:
            shift_labels = target_ids[..., 1:].contiguous()
            new_num_loss_tokens = (shift_labels != -100).sum().item()

            # if new_num_loss_tokens != num_loss_tokens:
            #     print(f'updating num_loss_tokens to {new_num_loss_tokens}, was {num_loss_tokens}')

            num_loss_tokens = new_num_loss_tokens

        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)

    print(ppl.item())


if __name__ == '__main__':
    main()
