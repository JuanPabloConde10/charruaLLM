"""Generate text from a trained charruaLLM checkpoint."""

import argparse

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def autodetect_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--prompt", default="El gobierno")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = autodetect_device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    if not ids:
        eot = tok.token_to_id("<|endoftext|>")
        ids = [eot if eot is not None else 0]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for i in range(args.num_samples):
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tok.decode(out[0].tolist())
        print("=" * 60)
        print(f"Sample {i+1}/{args.num_samples}")
        print("=" * 60)
        print(text)


if __name__ == "__main__":
    main()
