"""Sample ~3GB from spanish-corpora/preprocessed, keeping each source in its own _sampled file."""

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in tqdm(f, desc=f"  counting {path.name}", unit=" lines", leave=False))


def sample_lines(src: Path, dst: Path, n_sample: int, seed: int):
    rng = random.Random(seed)
    total = count_lines(src)
    indices = set(rng.sample(range(total), n_sample))
    
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for i, line in tqdm(enumerate(fin), total=total, desc=f"  sampling {src.name}", unit=" lines", leave=False):
            if i in indices:
                fout.write(line)
                written += 1
                if written % 100000 == 0:
                    sys.stdout.write(f"    [{written:,} lines written]\n")
                    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", default="spanish-corpora/preprocessed")
    p.add_argument("--target-gb", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-subdir", default="small-sample")
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = src_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)
    
    files = sorted(src_dir.glob("preprocessed_*_lower.txt"))
    assert files, f"no matching files in {src_dir}"

    total_bytes = sum(f.stat().st_size for f in files)
    total_gb = total_bytes / (1024 ** 3)
    frac = args.target_gb / total_gb
    print(f"Total corpus: {total_gb:.1f} GB across {len(files)} files")
    print(f"Target: {args.target_gb} GB -> sampling fraction: {frac:.4f} ({frac*100:.2f}%)")

    total_sampled = 0
    for f in files:
        n_lines = count_lines(f)
        n_sample = max(1, int(n_lines * frac))
        dst = out_dir / f.name.replace("_lower.txt", "_lower_sampled.txt")

        print(f"  {f.name}: {n_lines:,} lines -> {n_sample:,} sampled -> {out_dir.name}/{dst.name}")
        sample_lines(f, dst, n_sample, args.seed)
        sampled_bytes = dst.stat().st_size
        total_sampled += sampled_bytes
        print(f"    wrote {sampled_bytes / (1024**2):.1f} MB")

    print(f"\nTotal sampled: {total_sampled / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
