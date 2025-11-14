#!/usr/bin/env python

import argparse
import torch
from transformers import AutoModelForCausalLM


def compute_sparsity(model):
    """
    Returns a dict:
        name -> (num_zeros, total_elements, sparsity_fraction)
    """
    stats = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.numel() == 0:
                continue
            tensor = param.detach()
            num_zeros = (tensor == 0).sum().item()
            total = tensor.numel()
            sparsity = num_zeros / total
            stats[name] = (num_zeros, total, sparsity)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pruned_model_name", type=str, default='TuKoResearch/ConnectomeGPT100M',
        help="HuggingFace id or local path for the pruned model",
    )
    parser.add_argument(
        "-d", "--dense_model_name", type=str, default='TuKoResearch/NoConnectomeGPT100M',
        help="HuggingFace id or local path for the dense model",
    )
    args = parser.parse_args()

    # Same loading style as evals/hellaswag.py
    pruned_model = AutoModelForCausalLM.from_pretrained(
        args.pruned_model_name, trust_remote_code=True
    )
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.dense_model_name, trust_remote_code=True
    )

    pruned_stats = compute_sparsity(pruned_model)
    dense_stats = compute_sparsity(dense_model)

    # union of parameter names from both models
    all_names = sorted(set(pruned_stats.keys()) | set(dense_stats.keys()))

    print("PRUNED! Parameter name".ljust(60), "Shape")
    print("-" * 80)

    for name, param in pruned_model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")


    print("name".ljust(60), "pruned_sparsity", "dense_sparsity")
    for name in all_names:
        p = pruned_stats.get(name)
        d = dense_stats.get(name)
        p_s = p[2] if p is not None else float("nan")
        d_s = d[2] if d is not None else float("nan")
        print(f"{name:60s} {p_s:10.4%} {d_s:10.4%}")

    # If you want the raw data in a list instead of printing:
    # pruned_list = [(n, *vals) for n, vals in pruned_stats.items()]
    # dense_list  = [(n, *vals) for n, vals in dense_stats.items()]



# 73.7852% sparsity for all parameters in the pruned model
# 0.0000% sparsity for all parameters in the dense model