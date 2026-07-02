"""Tokenize the .txt corpus into train.bin / val.bin (numpy uint16) + meta.json."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--val-frac", type=float, default=0.005, help="fraction of lines held out for validation")
    p.add_argument("--batch-lines", type=int, default=4096, help="lines tokenized per batch (HF tokenizers is parallel)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None, "tokenizer missing <|endoftext|>"
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size < (1 << 16) else np.uint32
    print(f"vocab_size={vocab_size}, dtype={dtype.__name__}, eot_id={eot_id}")

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.txt"))
    assert files, f"no .txt files in {data_dir}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    rng = np.random.default_rng(args.seed)
    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")
    train_count = 0
    val_count = 0

    def flush(buf):
        nonlocal train_count, val_count
        if not buf:
            return
        encs = tok.encode_batch(buf)
        train_ids = []
        val_ids = []
        for e in encs:
            ids = e.ids
            if rng.random() < args.val_frac:
                val_ids.extend(ids)
            else:
                train_ids.extend(ids)
        if train_ids:
            arr = np.asarray(train_ids, dtype=dtype)
            arr.tofile(train_f)
            train_count += len(arr)
        if val_ids:
            arr = np.asarray(val_ids, dtype=dtype)
            arr.tofile(val_f)
            val_count += len(arr)

    for path in files:
        print(f"\nTokenizing {path.name}")
        buf = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, desc=path.name, unit="lines"):
                line = line.strip()
                if not line:
                    continue
                buf.append(line)
                if len(buf) >= args.batch_lines:
                    flush(buf)
                    buf = []
            flush(buf)
        # one EOT between files to mark a soft "document" boundary
        np.asarray([eot_id], dtype=dtype).tofile(train_f)
        train_count += 1

    train_f.close()
    val_f.close()

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "eot_id": eot_id,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "files": [f.name for f in files],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {train_path}: {train_count:,} tokens")
    print(f"Wrote {val_path}:   {val_count:,} tokens")
    print(f"Wrote {out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
