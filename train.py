"""Training loop for charruaLLM base model."""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def autodetect_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def estimate_loss(model, data_dict, block_size, batch_size, device, n_eval=50):
    model.eval()
    out = {}
    for split, data in data_dict.items():
        ls = []
        for _ in range(n_eval):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            ls.append(loss.item())
        out[split] = sum(ls) / len(ls)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out-dir", default="runs/run1")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
    # Model
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    # Optim
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Logging / IO
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = autodetect_device(args.device)
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    # mixed precision
    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32  # MPS / CPU stay in fp32
    print(f"Compute dtype: {amp_dtype}")

    # data
    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    np_dtype = getattr(np, meta["dtype"])
    train_data = np.memmap(data_dir / "train.bin", dtype=np_dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np_dtype, mode="r")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")
    data_dict = {"train": train_data, "val": val_data}

    # model
    cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Model params: {model.num_params():,}")

    raw_model = model
    if args.compile:
        if device == "cuda":
            print("torch.compile enabled")
            model = torch.compile(model)
        else:
            print("torch.compile requested but device != cuda; skipping")

    # optimizer (AdamW, fused on CUDA)
    optim_kwargs = dict(lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if device == "cuda":
        optim_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "model_cfg": cfg.__dict__, "meta": meta}, f, indent=2)

    autocast_ctx = (
        torch.amp.autocast(device_type=device, dtype=amp_dtype)
        if amp_dtype != torch.float32
        else nullcontext()
    )

    t0 = time.time()
    best_val = float("inf")

    for step in range(args.max_steps + 1):
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % args.eval_every == 0:
            losses = estimate_loss(model, data_dict, args.block_size, args.batch_size, device, n_eval=args.eval_iters)
            elapsed = (time.time() - t0) / 60
            print(
                f"[eval] step {step:6d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}m"
            )
            if losses["val"] < best_val and step > 0:
                best_val = losses["val"]
                ckpt = {
                    "model": raw_model.state_dict(),
                    "cfg": cfg.__dict__,
                    "step": step,
                    "val_loss": losses["val"],
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"       saved best.pt (val={best_val:.4f})")

        if step == args.max_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            last_loss += loss.item()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            print(f"step {step:6d} | loss {last_loss:.4f} | lr {lr:.2e}")

    torch.save(
        {"model": raw_model.state_dict(), "cfg": cfg.__dict__, "step": args.max_steps},
        out_dir / "final.pt",
    )
    print(f"Saved {out_dir/'final.pt'}  | best val {best_val:.4f}")


if __name__ == "__main__":
    main()
