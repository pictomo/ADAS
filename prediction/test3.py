from pathlib import Path

from util.data import build_entity_lookup, load_data, make_text
from util.vec import cos_sim, dist, str2vec


def pick_pairs(
    abt_rows: list[dict[str, str]],
    buy_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
):
    """正例3組と負例3組を、IDの重複がないように選ぶ。"""
    match_set = {(int(row["idAbt"]), int(row["idBuy"])) for row in mapping_rows}

    matched_pairs = []
    used_abt_ids = set()
    used_buy_ids = set()

    # 正解対応表の先頭から、Abt 側と Buy 側が重ならない正例を3組取る。
    for row in mapping_rows:
        abt_id = int(row["idAbt"])
        buy_id = int(row["idBuy"])
        if abt_id in used_abt_ids or buy_id in used_buy_ids:
            continue

        matched_pairs.append((abt_id, buy_id))
        used_abt_ids.add(abt_id)
        used_buy_ids.add(buy_id)

        if len(matched_pairs) == 3:
            break

    if len(matched_pairs) < 3:
        raise ValueError("Could not find 3 disjoint matching pairs")

    negative_pairs = []
    # 正例で使った ID を避けつつ、対応表に存在しない組を3組選ぶ。
    remaining_abt_ids = [
        int(row["id"]) for row in abt_rows if int(row["id"]) not in used_abt_ids
    ]
    remaining_buy_ids = [
        int(row["id"]) for row in buy_rows if int(row["id"]) not in used_buy_ids
    ]

    for abt_id in remaining_abt_ids:
        for buy_id in remaining_buy_ids:
            if (abt_id, buy_id) in match_set:
                continue

            negative_pairs.append((abt_id, buy_id))
            used_abt_ids.add(abt_id)
            used_buy_ids.add(buy_id)

            if len(negative_pairs) == 3:
                break
        if len(negative_pairs) == 3:
            break

    if len(negative_pairs) < 3:
        raise ValueError("Could not find 3 disjoint non-matching pairs")

    return matched_pairs, negative_pairs


if __name__ == "__main__":
    # データファイルの場所を確定して、以降の処理では中身だけを扱う。
    root = Path(__file__).resolve().parents[1]
    abt_path = root / "dataset" / "Abt-Buy" / "Abt.csv"
    buy_path = root / "dataset" / "Abt-Buy" / "Buy.csv"
    mapping_path = root / "dataset" / "Abt-Buy" / "abt_buy_perfectMapping.csv"

    # 各 CSV を独立に読み込み、表ごとの行一覧を用意する。
    abt_rows = load_data(abt_path)
    buy_rows = load_data(buy_path)
    mapping_rows = load_data(mapping_path)

    # 後続の比較処理で何度も参照するため、ID -> 行データの辞書にしておく。
    abt_lookup = build_entity_lookup(abt_rows)
    buy_lookup = build_entity_lookup(buy_rows)

    # 正例3件と負例3件を選び、後で見やすい順番に並べ直す。
    matched_pairs, negative_pairs = pick_pairs(abt_rows, buy_rows, mapping_rows)
    all_pairs = [("match", pair) for pair in matched_pairs] + [
        ("non_match", pair) for pair in negative_pairs
    ]

    # ペアごとの指標を、正例と負例で分けて蓄積する。
    match_results: list[tuple[float, float]] = []
    nonmatch_results: list[tuple[float, float]] = []

    # 各ペアについて、実際に埋め込み化して距離とコサイン類似度を計算する。
    for label, (abt_id, buy_id) in all_pairs:
        abt_row = abt_lookup[abt_id]
        buy_row = buy_lookup[buy_id]

        abt_vec = str2vec(make_text(abt_row))
        buy_vec = str2vec(make_text(buy_row))

        d = dist(abt_vec, buy_vec)
        cs = cos_sim(abt_vec, buy_vec)

        if label == "match":
            match_results.append((d, cs))
        else:
            nonmatch_results.append((d, cs))

    # 出力は最小限にして、比較値だけを見られるようにする。
    print("MATCH")
    for d, cs in match_results:
        print(f"{d:.6f}, {cs:.6f}")

    print("NON_MATCH")
    for d, cs in nonmatch_results:
        print(f"{d:.6f}, {cs:.6f}")
