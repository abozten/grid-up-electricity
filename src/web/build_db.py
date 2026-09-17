"""
ETL script to build the SQLite database for the Grid-Up Transformer Web Explorer and Analysis.

Ingests all competition data, weather indicators, EPİAŞ external grid data, submissions,
district aggregates, and experiment result metrics into data/transformers.db.
"""

import os
import sys
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_sqlite_db import build_all_sqlite_database


def build_database(db_path: str = "data/transformers.db") -> None:
    build_all_sqlite_database(db_path)


if __name__ == "__main__":
    db_out = sys.argv[1] if len(sys.argv) > 1 else "data/transformers.db"
    build_database(db_out)
