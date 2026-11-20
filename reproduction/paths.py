from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction" / "records"
OUTPUT = ROOT / "build" / "paper"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
