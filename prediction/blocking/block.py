"""
Blocking: AbtとBuyの全商品をベクトル化し、近傍探索で同一商品候補のペアを絞り込む。

アルゴリズム概要:
  1. 両テーブルの全商品をOpenAI Embeddingでベクトル化
  2. FAISSのIndexFlatL2（完全探索L2距離）でインデックスを構築
  3. 双方向の近傍探索でペアを収集し重複を除外
     - 各Abt商品 → Buyから上位TOP_K件
     - 各Buy商品 → Abtから上位TOP_K件
  4. 結果を (idAbt, idBuy) 形式のCSVに出力
"""

import csv
import sys
from pathlib import Path

import faiss
import numpy as np

# プロジェクトルート (ADAS/) を sys.path に追加して prediction パッケージを解決する
sys.path.append(str(Path(__file__).resolve().parents[2]))

from prediction.embedding.util.data import load_data, make_text
from prediction.embedding.util.vec import str2vec

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "Abt-Buy"
OUTPUT_PATH = DATASET_DIR / "abt_buy_candidates.csv"
TOP_K = 10


def vectorize_rows(rows: list[dict[str, str]], label: str) -> np.ndarray:
    """全行をテキスト化してベクトル配列に変換する。str2vec はキャッシュを持つ。"""
    vecs = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        vecs.append(str2vec(make_text(row)))
        if i % 100 == 0 or i == n:
            print(f"  {label}: {i}/{n}")
    return np.stack(vecs).astype(np.float32)


def main() -> None:
    # --- データ読み込み ---
    abt_rows = load_data(DATASET_DIR / "Abt.csv")
    buy_rows = load_data(DATASET_DIR / "Buy.csv")

    # position → 元テーブルID の対応表（FAISSはインデックス上の位置を返すため）
    abt_ids = [int(row["id"]) for row in abt_rows]
    buy_ids = [int(row["id"]) for row in buy_rows]

    # --- ベクトル化 ---
    # APIコールはキャッシュ済みなら省略されるため2回目以降は即時完了
    print("Vectorizing Abt products...")
    abt_vecs = vectorize_rows(abt_rows, "Abt")
    print("Vectorizing Buy products...")
    buy_vecs = vectorize_rows(buy_rows, "Buy")

    dim = abt_vecs.shape[1]  # text-embedding-3-small は 1536次元

    # --- FAISSインデックス構築 ---
    # IndexFlatL2: 近似なしの完全探索。商品数が~1000件なので速度は問題ない
    buy_index = faiss.IndexFlatL2(dim)
    buy_index.add(buy_vecs)

    abt_index = faiss.IndexFlatL2(dim)
    abt_index.add(abt_vecs)

    # --- 双方向近傍探索 ---
    candidates: set[tuple[int, int]] = set()

    # Abt→Buy: 各Abt商品に対してBuyから上位TOP_K件を取得
    _, buy_nn = buy_index.search(abt_vecs, TOP_K)
    for abt_pos, nn in enumerate(buy_nn):
        abt_id = abt_ids[abt_pos]
        for buy_pos in nn:
            if buy_pos >= 0:  # FAISSはベクトル数 < k の場合に -1 を返す
                candidates.add((abt_id, buy_ids[buy_pos]))

    # Buy→Abt: 各Buy商品に対してAbtから上位TOP_K件を取得
    _, abt_nn = abt_index.search(buy_vecs, TOP_K)
    for buy_pos, nn in enumerate(abt_nn):
        buy_id = buy_ids[buy_pos]
        for abt_pos in nn:
            if abt_pos >= 0:
                candidates.add((abt_ids[abt_pos], buy_id))

    print(f"Total candidate pairs: {len(candidates)}")

    # --- 結果出力 ---
    # sorted でソートして出力の再現性を確保
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idAbt", "idBuy"])
        for abt_id, buy_id in sorted(candidates):
            writer.writerow([abt_id, buy_id])

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
