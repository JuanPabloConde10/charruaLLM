"""Prepare a large Spanish text corpus without loading it into RAM.

Input files are UTF-8 ``.txt`` files. Non-empty consecutive lines form one
document and blank lines separate documents, matching ``prepare_data.py``.

Unlike the original preparer, this implementation:

* streams the corpus in a single pass;
* assigns whole documents to train/validation/test deterministically;
* splits exceptionally large documents into bounded tokenizer inputs;
* tokenizes bounded batches in parallel through Hugging Face Tokenizers; and
* writes token IDs incrementally to disk.

The resulting ``train.bin``, ``val.bin``, ``test.bin`` and ``meta.json`` are
directly compatible with ``train.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from tqdm import tqdm


SPLITS = ("train", "val", "test")
MASK_64 = (1 << 64) - 1


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def splitmix64(value: int) -> int:
    """Return a stable pseudo-random uint64 derived from ``value``."""
    value = (value + 0x9E3779B97F4A7C15) & MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_64
    return (value ^ (value >> 31)) & MASK_64


def choose_split(
    document_index: int,
    seed: int,
    val_threshold: int,
    test_threshold: int,
) -> str:
    """Assign a document to a split without storing or pre-counting documents."""
    draw = splitmix64(document_index ^ (seed & MASK_64))
    if draw < val_threshold:
        return "val"
    if draw < test_threshold:
        return "test"
    return "train"


def iter_document_chunks(
    files: list[Path],
    chunk_chars: int,
    progress: tqdm,
) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(document_index, text_chunk, is_last_chunk)`` in corpus order.

    Chunk boundaries prefer whitespace and preserve the exact normalized text
    (stripped lines joined by one space). Only tokenization merges crossing a
    chunk boundary can differ from whole-document tokenization.
    """
    document_index = 0
    in_document = False
    parts: list[str] = []
    parts_length = 0
    min_boundary_offset = max(1, int(chunk_chars * 0.8))

    for path in files:
        with path.open("rb") as handle:
            for raw_line in handle:
                progress.update(len(raw_line))
                line = raw_line.decode("utf-8").strip()

                if not line:
                    if in_document:
                        final_chunk = "".join(parts)
                        if final_chunk:
                            yield document_index, final_chunk, True
                        document_index += 1
                        in_document = False
                        parts = []
                        parts_length = 0
                    continue

                piece = line if not in_document else " " + line
                in_document = True
                parts.append(piece)
                parts_length += len(piece)

                if parts_length <= chunk_chars:
                    continue

                combined = "".join(parts)
                start = 0
                combined_length = len(combined)

                while combined_length - start > chunk_chars:
                    preferred_start = start + min_boundary_offset
                    hard_end = start + chunk_chars
                    cut = combined.rfind(" ", preferred_start, hard_end + 1)
                    if cut < preferred_start:
                        cut = hard_end
                    yield document_index, combined[start:cut], False
                    start = cut

                remainder = combined[start:]
                parts = [remainder] if remainder else []
                parts_length = len(remainder)

        # A document never spans source files, matching collect_documents().
        if in_document:
            final_chunk = "".join(parts)
            if final_chunk:
                yield document_index, final_chunk, True
            document_index += 1
            in_document = False
            parts = []
            parts_length = 0


class IncrementalSplitWriter:
    """Write native-endian NumPy IDs and calculate checksums as data arrives."""

    def __init__(self, paths: dict[str, Path], dtype: type[np.generic], eos_id: int):
        self.paths = paths
        self.dtype = dtype
        self.handles: dict[str, BinaryIO] = {
            split: path.open("wb", buffering=0) for split, path in paths.items()
        }
        self.hashes = {split: hashlib.sha256() for split in SPLITS}
        self.token_counts = {split: 0 for split in SPLITS}
        self.document_counts = {split: 0 for split in SPLITS}
        self.chunk_counts = {split: 0 for split in SPLITS}
        self.eos_bytes = np.asarray([eos_id], dtype=dtype).tobytes()

    def write_encoding(self, split: str, ids: list[int], end_document: bool) -> None:
        payload = np.asarray(ids, dtype=self.dtype).tobytes()
        if payload:
            self.handles[split].write(payload)
            self.hashes[split].update(payload)
            self.token_counts[split] += len(ids)

        self.chunk_counts[split] += 1
        if end_document:
            self.handles[split].write(self.eos_bytes)
            self.hashes[split].update(self.eos_bytes)
            self.token_counts[split] += 1
            self.document_counts[split] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            if not handle.closed:
                handle.close()

    def checksums(self) -> dict[str, str]:
        return {split: digest.hexdigest() for split, digest in self.hashes.items()}


def detect_worker_count(requested: int) -> int:
    if requested > 0:
        return requested

    for variable in ("RAYON_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(variable, "")
        try:
            workers = int(value)
        except ValueError:
            continue
        if workers > 0:
            return workers

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def verify_outputs(out_dir: Path) -> bool:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        return False

    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)

    all_ok = True
    for split in SPLITS:
        path = out_dir / f"{split}.bin"
        expected = meta.get(f"{split}_sha256")
        if expected is None:
            print(f"  {split}: no checksum in meta.json (skipped)")
            continue
        if not path.exists():
            print(f"  {split}: {path} not found")
            all_ok = False
            continue
        actual = sha256_file(path)
        if actual == expected:
            print(f"  {split}: OK ({actual[:16]}...)")
        else:
            print(f"  {split}: MISMATCH")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
            all_ok = False
    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and tokenize a large Spanish corpus with bounded RAM usage."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--out-dir", default="processed")
    parser.add_argument("--val-frac", type=float, default=0.005)
    parser.add_argument("--test-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=128_000,
        help="maximum tokenizer input size per chunk (default: 128000)",
    )
    parser.add_argument(
        "--batch-chunks",
        "--batch-docs",
        dest="batch_chunks",
        type=int,
        default=256,
        help="maximum chunks sent to the tokenizer per batch (default: 256)",
    )
    parser.add_argument(
        "--batch-chars",
        type=int,
        default=4_000_000,
        help="maximum total characters per tokenizer batch (default: 4000000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="tokenizer threads; 0 uses Slurm allocation/CPU affinity (default: 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace existing train/val/test/meta outputs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify existing outputs against checksums in meta.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.val_frac < 1:
        raise SystemExit("--val-frac must be in [0, 1)")
    if not 0 <= args.test_frac < 1:
        raise SystemExit("--test-frac must be in [0, 1)")
    if args.val_frac + args.test_frac >= 1:
        raise SystemExit("--val-frac + --test-frac must be less than 1")
    if args.chunk_chars <= 0:
        raise SystemExit("--chunk-chars must be positive")
    if args.batch_chunks <= 0:
        raise SystemExit("--batch-chunks must be positive")
    if args.batch_chars < args.chunk_chars:
        raise SystemExit("--batch-chars must be greater than or equal to --chunk-chars")
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    out_dir = Path(args.out_dir)
    if args.verify:
        print("Verifying checksums...")
        if verify_outputs(out_dir):
            print("All checksums verified.")
        else:
            raise SystemExit("Some checksums failed verification.")
        return

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {data_dir}")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")

    workers = detect_worker_count(args.workers)
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Import only after configuring Rayon's global worker pool.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token = "<|endoftext|>"
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise SystemExit(f"tokenizer missing {eos_token!r}")

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= (1 << 16) else np.uint32
    total_source_bytes = sum(path.stat().st_size for path in files)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {split: out_dir / f"{split}.bin" for split in SPLITS}
    meta_path = out_dir / "meta.json"
    existing = [path for path in [*final_paths.values(), meta_path] if path.exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"output files already exist in {out_dir}: {names}; "
            "use --force or choose another --out-dir"
        )

    process_id = os.getpid()
    temporary_paths = {
        split: out_dir / f".{split}.bin.tmp.{process_id}" for split in SPLITS
    }
    temporary_meta_path = out_dir / f".meta.json.tmp.{process_id}"

    print(
        f"vocab_size={vocab_size}, dtype={dtype.__name__}, eos_id={eos_id}, "
        f"workers={workers}"
    )
    print(
        f"Input: {len(files)} files, {total_source_bytes / (1024 ** 3):.2f} GiB | "
        f"chunk_chars={args.chunk_chars:,}, batch_chars={args.batch_chars:,}"
    )
    print("Split assignment: deterministic single-pass hash by document")

    val_threshold = int(args.val_frac * (1 << 64))
    test_threshold = int((args.val_frac + args.test_frac) * (1 << 64))
    writer: IncrementalSplitWriter | None = None
    started_at = time.perf_counter()

    try:
        writer = IncrementalSplitWriter(temporary_paths, dtype, eos_id)
        batch: list[tuple[str, str, bool]] = []
        batch_character_count = 0
        encode_batch = getattr(tokenizer, "encode_batch_fast", tokenizer.encode_batch)

        with tqdm(
            total=total_source_bytes,
            desc="reading/tokenizing",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            smoothing=0.1,
        ) as progress:

            def flush_batch() -> None:
                nonlocal batch, batch_character_count
                if not batch:
                    return
                encodings = encode_batch(
                    [text for _, text, _ in batch],
                    add_special_tokens=False,
                )
                for (split, _, end_document), encoding in zip(batch, encodings):
                    writer.write_encoding(split, encoding.ids, end_document)
                batch = []
                batch_character_count = 0
                progress.set_postfix(
                    docs=f"{sum(writer.document_counts.values()):,}",
                    tokens=f"{sum(writer.token_counts.values()):,}",
                    refresh=False,
                )

            for document_index, text, end_document in iter_document_chunks(
                files, args.chunk_chars, progress
            ):
                split = choose_split(
                    document_index,
                    args.seed,
                    val_threshold,
                    test_threshold,
                )
                if batch and (
                    len(batch) >= args.batch_chunks
                    or batch_character_count + len(text) > args.batch_chars
                ):
                    flush_batch()

                batch.append((split, text, end_document))
                batch_character_count += len(text)

                if (
                    len(batch) >= args.batch_chunks
                    or batch_character_count >= args.batch_chars
                ):
                    flush_batch()

            flush_batch()

        writer.close()
        checksums = writer.checksums()
        elapsed_seconds = time.perf_counter() - started_at
        total_tokens = sum(writer.token_counts.values())

        meta = {
            "vocab_size": vocab_size,
            "dtype": dtype.__name__,
            "eos_token": eos_token,
            "eos_id": eos_id,
            "split_unit": "document",
            "split_method": "splitmix64_document_index",
            "document_separator": "blank_line",
            "line_joiner": "single_space",
            "tokenization_mode": "bounded_streaming_chunks",
            "chunk_chars": args.chunk_chars,
            "batch_chunks": args.batch_chunks,
            "batch_chars": args.batch_chars,
            "workers": workers,
            "train_documents": writer.document_counts["train"],
            "val_documents": writer.document_counts["val"],
            "test_documents": writer.document_counts["test"],
            "train_chunks": writer.chunk_counts["train"],
            "val_chunks": writer.chunk_counts["val"],
            "test_chunks": writer.chunk_counts["test"],
            "train_tokens": writer.token_counts["train"],
            "val_tokens": writer.token_counts["val"],
            "test_tokens": writer.token_counts["test"],
            "train_sha256": checksums["train"],
            "val_sha256": checksums["val"],
            "test_sha256": checksums["test"],
            "files": [path.name for path in files],
            "source_bytes": total_source_bytes,
            "seed": args.seed,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "elapsed_seconds": elapsed_seconds,
        }
        with temporary_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")

        for split in SPLITS:
            os.replace(temporary_paths[split], final_paths[split])
        os.replace(temporary_meta_path, meta_path)

    except BaseException:
        if writer is not None:
            writer.close()
        for path in [*temporary_paths.values(), temporary_meta_path]:
            path.unlink(missing_ok=True)
        raise

    elapsed_minutes = elapsed_seconds / 60
    rate = total_tokens / elapsed_seconds if elapsed_seconds else 0
    print()
    for split in SPLITS:
        print(
            f"{split:>5}: {writer.document_counts[split]:,} documents | "
            f"{writer.token_counts[split]:,} tokens | "
            f"sha256={checksums[split][:16]}..."
        )
    print(
        f"Wrote {out_dir} in {elapsed_minutes:.1f} min "
        f"({rate:,.0f} tokens/s) and {meta_path}"
    )


if __name__ == "__main__":
    main()
