"""Validasi dan persiapan dataset StudentDOA sebelum pemodelan."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "studentDOA.csv"
OUTPUT_FILE = BASE_DIR / "studentDOA_prepared.csv"
TARGET = "Target"


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {SOURCE_FILE}")

    data = pd.read_csv(SOURCE_FILE)
    if TARGET not in data.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")

    # Hapus baris duplikat dan baris tanpa target; fitur kosong dibiarkan agar
    # tidak menghapus banyak data secara diam-diam.
    before = len(data)
    data = data.drop_duplicates().dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET] != ""].reset_index(drop=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Data awal: {before:,} baris")
    print(f"Data siap: {len(data):,} baris, {len(data.columns)} kolom")
    print(f"Tersimpan: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
