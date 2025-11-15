import argparse
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

"""
Take a first stab at analyzing the connections of the GPT models (pruned/random pruned/dense).

For a GPT model, we have four matrices for each layer, e.g. for layer (i.e. block) 1:

transformer.h.1.attn.c_attn.weight                           (2304, 768)    OBS, we could analyze this as 3 x (768, 768) for Q,K,V separately
transformer.h.1.attn.c_proj.weight                           (768, 768)
transformer.h.1.mlp.c_fc.weight                              (3072, 768)
transformer.h.1.mlp.c_proj.weight                            (768, 3072)

.... etc.

Abstracted away, this gives:

attn.c_attn.weight  : residual -> [Q,K,V]
attn.c_proj.weight  : attention output -> residual
mlp.c_fc.weight     : residual -> MLP hidden
mlp.c_proj.weight   : MLP hidden -> residual


For a weight matrix W of shape (out_dim, in_dim), the element W[i,j] represents a synapse from
input unit j to output unit i.
We can analyze the "connectome" of a given layer by looking at the in-degrees and out-degrees of the residual nodes,
i.e., the number of incoming and outgoing connections to/from each residual channel (of size d_model, in this case 768).

"""


def analyze_single_layer(model, layer_idx: int = 1, zero_tol: float = 0.0):
    """
    Analyze one transformer layer's 'connectome' for residual nodes.

    Definitions
    -----------
    - residual channel (c):
        index into the residual vector x ∈ R^{d_model} at this layer.

    - residual node R_c:
        the scalar unit corresponding to residual channel c.

    - out-degree(R_c):
        number of nonzero outgoing synapses from R_c to:
          * all Q,K,V units (via attn.c_attn)      --> this is the transformation from residual to QKV
          * all MLP hidden units (via mlp.c_fc)

        Implemented as:
          - count nonzero entries in column c of attn.c_attn and mlp.c_fc.

    - in-degree(R_c):
        number of nonzero incoming synapses into R_c from:
          * attention output units (via attn.c_proj)    --> this is the transformation from attn output to residual
          * MLP hidden units (via mlp.c_proj)

        Implemented as:
          - count nonzero entries in row c of attn.c_proj and mlp.c_proj.

    - strengths:
        out-strength(R_c) = sum of |weights| on those outgoing edges
                            (same columns, masked by zero_tol).

        in-strength(R_c)  = sum of |weights| on those incoming edges
                            (same rows, masked by zero_tol).

        total_strength(R_c) = in-strength + out-strength

    """
    block = model.transformer.h[layer_idx]

    # The four connection matrices for this layer
    W_attn_in = block.attn.c_attn.weight.detach()   # [3*d_model, d_model]
    W_attn_out = block.attn.c_proj.weight.detach()  # [d_model, d_model]
    W_mlp_in = block.mlp.c_fc.weight.detach()       # [d_mlp, d_model]
    W_mlp_out = block.mlp.c_proj.weight.detach()    # [d_model, d_mlp]

    d_model = W_attn_out.shape[0]  # number of residual channels, in this case 768

    print(f"Analyzing layer {layer_idx}")
    print(f"  attn.c_attn.weight shape: {tuple(W_attn_in.shape)}")
    print(f"  attn.c_proj.weight  shape: {tuple(W_attn_out.shape)}")
    print(f"  mlp.c_fc.weight     shape: {tuple(W_mlp_in.shape)}")
    print(f"  mlp.c_proj.weight   shape: {tuple(W_mlp_out.shape)}")

    # Count nonzero entries
    mask_attn_in = (W_attn_in.abs() > zero_tol)    # residual -> QKV
    mask_attn_out = (W_attn_out.abs() > zero_tol)  # attn -> residual
    mask_mlp_in = (W_mlp_in.abs() > zero_tol)      # residual -> MLP
    mask_mlp_out = (W_mlp_out.abs() > zero_tol)    # MLP -> residual

    # --- Degrees ---

    # Out-degree: count nonzero entries in columns where residual is input.
    # Columns = residual channels c.
    out_deg_attn = mask_attn_in.sum(dim=0)   # [d_model]
    out_deg_mlp = mask_mlp_in.sum(dim=0)     # [d_model]
    out_degree = out_deg_attn + out_deg_mlp  # [d_model]

    # In-degree: count nonzero entries in rows where residual is output.
    # Rows = residual channels c.
    in_deg_attn = mask_attn_out.sum(dim=1)   # [d_model]
    in_deg_mlp = mask_mlp_out.sum(dim=1)     # [d_model]
    in_degree = in_deg_attn + in_deg_mlp     # [d_model]

    total_degree = in_degree + out_degree
    # In principle we could also look at these ones separately??!

    # --- Strengths (sum of |weights| on the same edges) ---

    # Out-strength: sum |W| over those same columns,
    # but only where mask is True (so strengths align with degrees).
    out_str_attn = (W_attn_in.abs() * mask_attn_in).sum(dim=0)  # [d_model]
    out_str_mlp = (W_mlp_in.abs() * mask_mlp_in).sum(dim=0)     # [d_model]
    out_strength = out_str_attn + out_str_mlp                   # [d_model]

    # In-strength: sum |W| over those same rows.
    in_str_attn = (W_attn_out.abs() * mask_attn_out).sum(dim=1)  # [d_model]
    in_str_mlp = (W_mlp_out.abs() * mask_mlp_out).sum(dim=1)     # [d_model]
    in_strength = in_str_attn + in_str_mlp                       # [d_model]

    total_strength = in_strength + out_strength

    # --- Simple summary statistics ---

    def summarize(name, tensor):
        x = tensor.float()
        print(
            f"{name:15s} | mean={x.mean().item():.3f} "
            f"std={x.std().item():.3f} "
            f"min={x.min().item():.3f} "
            f"max={x.max().item():.3f}"
        )

    print("\nSummary over residual nodes in this layer:")
    summarize("in_degree", in_degree)
    summarize("out_degree", out_degree)
    summarize("total_degree", total_degree)
    summarize("in_strength", in_strength)
    summarize("out_strength", out_strength)
    summarize("total_strength", total_strength)

    # --- Per-node table (TSV) ---

    print("\nchannel\tin_degree\tout_degree\ttotal_degree\tin_strength\tout_strength\ttotal_strength")
    for c in range(d_model):
        print(
            f"{c}\t"
            f"{int(in_degree[c])}\t"
            f"{int(out_degree[c])}\t"
            f"{int(total_degree[c])}\t"
            f"{in_strength[c].item():.6e}\t"
            f"{out_strength[c].item():.6e}\t"
            f"{total_strength[c].item():.6e}"
        )

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "in_strength": in_strength,
        "out_strength": out_strength,
        "total_strength": total_strength,
    }


def hist_plots(stats, bins: int = 50):
    """
    Tiny helper to visualize the distributions:

    - Histogram of total_degree
    - Histogram of log10(total_strength) for nodes with strength > 0

    If matplotlib is not installed, this just prints a message and does nothing.
    """
    if plt is None:
        print("\n[hist_plots] matplotlib not available; skipping plots.")
        return

    total_degree = stats["total_degree"].detach().cpu()
    total_strength = stats["total_strength"].detach().cpu()

    # 1) Histogram of total_degree
    plt.figure()
    plt.hist(total_degree.numpy(), bins=bins)
    plt.title("Total degree per residual node")
    plt.xlabel("total_degree")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # 2) Histogram of log10(total_strength) (only for positive strengths)
    mask_pos = total_strength > 0
    if mask_pos.any():
        log_strength = torch.log10(total_strength[mask_pos])
        plt.figure()
        plt.hist(log_strength.numpy(), bins=bins)
        plt.title("log10(total strength) per residual node")
        plt.xlabel("log10(total_strength)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()
    else:
        print("[hist_plots] No positive strengths found; skipping log-strength plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        # default="TuKoResearch/NoConnectomeGPT100M", # DENSE MODEL
        # default="TuKoResearch/RandomConnectomeGPT100M", # RANDOM PRUNED MODEL
        default="TuKoResearch/ConnectomeGPT100M", # PRUNED MODEL
        help="HuggingFace id or local path for the model to analyze",
    )
    parser.add_argument(
        "-l",
        "--layer_idx",
        type=int,
        default=10,
        help="Transformer block index to analyze (e.g., 1 for transformer.h.1)",
    )
    parser.add_argument(
        "--zero_tol",
        type=float,
        default=0.0,
        help="Threshold below which weights are treated as zero.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, skip the quick histograms.",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    stats = analyze_single_layer(model, layer_idx=args.layer_idx, zero_tol=args.zero_tol)

    if not args.no_plots:
        hist_plots(stats)
