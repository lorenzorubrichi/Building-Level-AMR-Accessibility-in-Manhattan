from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

for path in [DATA_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print(f"Project initialized in: {BASE_DIR}")
