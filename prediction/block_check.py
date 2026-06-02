"""
Blocking の候補ペアが、正解対応表をどれだけ含んでいるかを確認する。

事前に block.py を実行して、dataset/Abt-Buy/abt_buy_candidates.csv を
生成しておくことを想定する。
"""

import csv
import sys
from pathlib import Path

# prediction/ を sys.path に追加して util パッケージを解決する
sys.path.append(str(Path(__file__).resolve().parent))

from util.data import load_data

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset" / "Abt-Buy"
CANDIDATES_PATH = DATASET_DIR / "abt_buy_candidates.csv"
MAPPING_PATH = DATASET_DIR / "abt_buy_perfectMapping.csv"


def load_candidate_pairs(path: Path) -> set[tuple[int, int]]:
    """block.py が出力した候補ペアを読み込む。"""
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {(int(row["idAbt"]), int(row["idBuy"])) for row in reader}


def main() -> None:
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"候補ファイルが見つかりません: {CANDIDATES_PATH}\n"
            "先に block.py を実行してください。"
        )

    candidates = load_candidate_pairs(CANDIDATES_PATH)
    mapping_rows = load_data(MAPPING_PATH)
    true_pairs = {(int(row["idAbt"]), int(row["idBuy"])) for row in mapping_rows}

    hits = sum(1 for pair in true_pairs if pair in candidates)
    total = len(true_pairs)
    percentage = (hits / total * 100.0) if total else 0.0

    print(f"hits: {hits}")
    print(f"total: {total}")
    print(f"percentage: {percentage:.2f}%")


if __name__ == "__main__":
    main()
