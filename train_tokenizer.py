"""Train a byte-level BPE tokenizer on a sample of the corpus."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="folder with .txt corpus files")
    p.add_argument("--out", default="tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=200_000_000,
        help="cap of bytes read per file (200MB by default keeps training fast)",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.txt"))
    assert files, f"No .txt files found in {data_dir}"
    print(f"Found {len(files)} corpus files in {data_dir}")

    def iter_lines():
        for f in files:
            print(f"  sampling from {f.name} (cap {args.max_bytes_per_file:,} bytes)")
            read = 0
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    read += len(line.encode("utf-8"))
                    if read > args.max_bytes_per_file:
                        break
                    line = line.strip()
                    if line:
                        yield line

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter_lines(), trainer=trainer)
    tokenizer.save(args.out)
    print(f"Saved {args.out} (vocab_size={tokenizer.get_vocab_size()})")

    sample = "El presidente uruguayo dijo que la inflación bajará el año próximo."
    enc = tokenizer.encode(sample)
    print(f"sanity check: {sample!r}")
    print(f"  {len(enc.ids)} tokens, first 10 ids = {enc.ids[:10]}")
    print(f"  decoded = {tokenizer.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
