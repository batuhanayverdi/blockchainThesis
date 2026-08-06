"""
Locate the Hive scrape archive and check existing files for needed columns
==========================================================================
Walks the Thesis folder (skipping virtual environments) and reports:
  1. any hive_full_posts_full.parquet (consolidated archive)
  2. any folder containing week*.parquet chunks (weekly archive)
  3. the columns of hive_sample_master.parquet and hive_cleaned.parquet,
     to see whether they already carry created / json_metadata / body,
     which would let hive_post_features.py read them directly instead.

Runs in well under a minute. Requirements: pandas, pyarrow (installed).
"""

import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(r"C:\Users\batuh\PycharmProjects\Thesis")
SKIP_DIRS = {".venv", ".venv_doc", ".git", "__pycache__", "node_modules",
             ".idea"}
NEEDED = {"created", "json_metadata", "body"}

CHECK_FILES = [
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet",
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_cleaned.parquet",
]


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main() -> None:
    print("=" * 72)
    print("Archive finder")
    print("=" * 72)

    consolidated, chunk_dirs = [], {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith(".parquet"):
                continue
            full = Path(dirpath) / f
            if "hive_full_posts_full" in f:
                consolidated.append(full)
            elif f.startswith("week") and "_to_" in f:
                chunk_dirs.setdefault(dirpath, []).append(full)

    print("\nConsolidated archive candidates:")
    if consolidated:
        for p in consolidated:
            print(f"  {p}   ({human(p.stat().st_size)})")
    else:
        print("  none found")

    print("\nWeekly chunk folders:")
    if chunk_dirs:
        for d, files in chunk_dirs.items():
            total = sum(f.stat().st_size for f in files)
            print(f"  {d}   ({len(files)} chunks, {human(total)})")
    else:
        print("  none found")

    print("\nColumn check on existing sample files:")
    for f in CHECK_FILES:
        p = Path(f)
        if not p.exists():
            print(f"  MISSING: {f}")
            continue
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        have = sorted(NEEDED & cols)
        lack = sorted(NEEDED - cols)
        print(f"  {p.name}: has {have if have else 'none of the needed'}"
              f"{',  lacks ' + str(lack) if lack else ''}")
        print(f"    all columns: {sorted(cols)}")

    print("\nSend this output back. If a sample file already has all three "
          "needed columns, hive_post_features.py will be pointed at it; "
          "otherwise set FULL_PARQUET or CHUNK_DIR to a path listed above.")


if __name__ == "__main__":
    main()
