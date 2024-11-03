"""Compute ESM2 PLL given binder sequence."""

import math
import torch
import esm
import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare multiple size models
model_650m, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_650m.eval()
model_650m = model_650m.to(device)

model_150m, _ = esm.pretrained.esm2_t30_150M_UR50D()
model_150m.eval()
model_150m = model_150m.to(device)

model_35m, _ = esm.pretrained.esm2_t12_35M_UR50D()
model_35m.eval()
model_35m = model_35m.to(device)

# alphabet is the same for all esm2 models
# print(alphabet.to_dict())
# {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
# 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
# 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
batch_converter = alphabet.get_batch_converter()


def compute_pll(sequence, model_type='650M'):
    model_type_lower = model_type.lower()
    if model_type_lower == '650m':
        model = model_650m
    elif model_type_lower == '150m':
        model = model_150m
    elif model_type_lower == '35m':
        model = model_35m
    else:
        raise ValueError(f'Model type must be 650M, 150M, or 35M: {model_type}')

    print(sequence, flush=True)

    data = [("protein", sequence)]
    batch_converter = alphabet.get_batch_converter()
    *_, batch_tokens = batch_converter(data)
    log_probs = []
    for i in range(len(sequence)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i + 1] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked.to(device))["logits"], dim=-1
            )
        log_probs.append(token_probs[0, i + 1, alphabet.get_idx(sequence[i])].item())
    return math.fsum(log_probs)


@click.command()
@click.argument("aa_sequence")
def main(aa_sequence):
    pll = compute_pll(aa_sequence)
    print(f"PLL: {pll}")


if __name__ == "__main__":
    main()
