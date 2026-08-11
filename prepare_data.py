"""Tokenize the .txt corpus by news article into train.bin / val.bin / test.bin + meta.json."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_documents(files: list[Path]) -> list[str]:
    documents = []
    current = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    current.append(line)
                    continue
                if current:
                    documents.append(" ".join(current))
                    current = []
        if current:
            documents.append(" ".join(current))
            current = []
    return documents


def tokenize_documents(tok: Tokenizer, documents: list[str], batch_size: int, eos_id: int) -> list[int]:
    all_ids = []
    for i in tqdm(range(0, len(documents), batch_size), desc="tokenizing", unit="batch"):
        batch = documents[i : i + batch_size]
        encs = tok.encode_batch(batch)
        for e in encs:
            all_ids.extend(e.ids)
            all_ids.append(eos_id)
    return all_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--val-frac", type=float, default=0.005, help="fraction of documents for validation")
    p.add_argument("--test-frac", type=float, default=0.005, help="fraction of documents for test")
    p.add_argument(
        "--batch-docs",
        "--batch-lines",
        dest="batch_docs",
        type=int,
        default=1024,
        help="documents tokenized per batch",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", action="store_true", help="verify checksums of existing .bin files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    test_path = out_dir / "test.bin"
    meta_path = out_dir / "meta.json"

    if args.verify:
        print("Verifying checksums...")
        if not meta_path.exists():
            print(f"ERROR: {meta_path} not found")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        all_ok = True
        for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            key = f"{split}_sha256"
            if key not in meta:
                print(f"  {split}: no checksum in meta.json (skipped)")
                continue
            if not path.exists():
                print(f"  {split}: {path} not found")
                all_ok = False
                continue
            actual = sha256_file(path)
            expected = meta[key]
            if actual == expected:
                print(f"  {split}: OK ({actual[:16]}...)")
            else:
                print(f"  {split}: MISMATCH")
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
                all_ok = False
        if all_ok:
            print("All checksums verified.")
        else:
            print("Some checksums failed verification.")
        return

    tok = Tokenizer.from_file(args.tokenizer)
    eos_token = "<|endoftext|>"
    eos_id = tok.token_to_id(eos_token)
    assert eos_id is not None, f"tokenizer missing {eos_token!r}"
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size < (1 << 16) else np.uint32
    print(f"vocab_size={vocab_size}, dtype={dtype.__name__}, eos_id={eos_id}")

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.txt"))
    assert files, f"no .txt files in {data_dir}"

    print("Collecting documents...")
    all_documents = collect_documents(files)
    total = len(all_documents)
    print(f"Total documents: {total:,}")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(total)

    n_val = int(total * args.val_frac)
    n_test = int(total * args.test_frac)
    val_indices = set(indices[:n_val].tolist())
    test_indices = set(indices[n_val:n_val + n_test].tolist())

    train_documents = [all_documents[i] for i in range(total) if i not in val_indices and i not in test_indices]
    val_documents = [all_documents[i] for i in indices[:n_val].tolist()]
    test_documents = [all_documents[i] for i in indices[n_val:n_val + n_test].tolist()]
    print(
        f"train: {len(train_documents):,} documents | "
        f"val: {len(val_documents):,} documents | "
        f"test: {len(test_documents):,} documents"
    )
    train_document_count = len(train_documents)
    val_document_count = len(val_documents)
    test_document_count = len(test_documents)

    del all_documents, indices

    def write_split(split_documents, path):
        ids = tokenize_documents(tok, split_documents, args.batch_docs, eos_id)
        arr = np.asarray(ids, dtype=dtype)
        arr.tofile(path)
        checksum = sha256_file(path)
        print(f"Wrote {path}: {len(arr):,} tokens (sha256={checksum[:16]}...)")
        return len(arr), checksum

    print("\nTokenizing train...")
    train_count, train_sha = write_split(train_documents, train_path)
    del train_documents

    print("\nTokenizing val...")
    val_count, val_sha = write_split(val_documents, val_path)
    del val_documents

    print("\nTokenizing test...")
    test_count, test_sha = write_split(test_documents, test_path)
    del test_documents

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "eos_token": eos_token,
        "eos_id": eos_id,
        "split_unit": "document",
        "document_separator": "blank_line",
        "train_documents": train_document_count,
        "val_documents": val_document_count,
        "test_documents": test_document_count,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "test_tokens": test_count,
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "files": [f.name for f in files],
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "batch_docs": args.batch_docs,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {meta_path}")


if __name__ == "__main__":
    main()
