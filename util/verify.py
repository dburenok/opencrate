import os
import csv
from pathlib import Path

# ---- Settings ----
INDEX_DIR = Path("../opencrate_index")
EMB_BIN = INDEX_DIR / "embeddings.bin"
MANIFEST = INDEX_DIR / "manifest.csv"
DIM = 512
# ------------------

def main():
    try:
        # Calculate number of embeddings from file size
        num_embeddings = os.path.getsize(EMB_BIN) // (DIM * 4)  # float32 = 4 bytes

        # Count rows in manifest
        with open(MANIFEST, "r", encoding="utf-8") as f:
            num_rows = sum(1 for _ in csv.reader(f))

    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found - {e.filename}\n")
        return

    # Output summary
    print(f"  Verifying Data Integrity in '{INDEX_DIR}'")
    print("  File Counts:")
    print(f"    - Embeddings:     {num_embeddings:,}")
    print(f"    - Manifest Rows:  {num_rows:,}\n")

    # Check counts match
    if len({num_embeddings, num_rows}) == 1:
        if num_embeddings > 0:
            print(f"  ✅ All counts match ({num_embeddings:,}).\n")
        else:
            print("  ⚠️ All files are empty. Counts match but contain no data.\n")
    else:
        print("  ❌ Counts do not match!\n")

if __name__ == "__main__":
    main()
