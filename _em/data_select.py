# AbtBuyデータセットからEM評価用データをサンプリングし、valとtestに分割するスクリプト
# testを先にサンプリング（自然分布を保持）し、残りからvalをbalancedサンプリングする
# 出力はabt_buy_candidates.csvと同形式のCSV（idAbt, idBuy, label）
# リポジトリルートから実行すること

import pandas as pd


# ===== データ読み込み =====

# 候補ペアと正解マッピングを読み込む
candidates = pd.read_csv("dataset/Abt-Buy/abt_buy_candidates.csv")
mapping = pd.read_csv("dataset/Abt-Buy/abt_buy_perfectMapping.csv")

# ===== ラベル付与 =====

# perfectMappingに含まれるペアをマッチ（True）として候補にラベルを付与
mapping_set = set(zip(mapping["idAbt"], mapping["idBuy"]))
candidates["label"] = candidates.apply(
    lambda r: (r["idAbt"], r["idBuy"]) in mapping_set, axis=1
)

total = len(candidates)
n_match = candidates["label"].sum()
n_non_match = total - n_match
print(f"Total candidates: {total}, Matches: {n_match}, Non-matches: {n_non_match}")

# ===== サンプリング =====

# testを先にサンプリングして自然分布（約7.5%マッチ率）を維持する
# valを先にサンプリングすると残りプールのマッチ率が下がるため、順序が重要
#
# 注意: reset_index() を残りプール計算の前に行ってはいけない。
# reset_index後は test_df.index = [0..999] となり、candidates.drop(test_df.index) が
# 本来のサンプル行でなく先頭1000行を除外してしまいval/testの重複が生じる。
test_df = candidates.sample(n=1000, random_state=0)
test_match = test_df["label"].sum()
print(f"Test: {len(test_df)} pairs, {test_match} matches ({test_match/len(test_df)*100:.1f}%)")

# testで使わなかった残りから、マッチ500件・非マッチ500件のbalanced valセットを作成
# test_df.index はオリジナルのcandidatesインデックスを保持したままdropする
remaining = candidates.drop(test_df.index)
remaining_matches = remaining[remaining["label"] == True]
remaining_non_matches = remaining[remaining["label"] == False]
print(f"Remaining pool: {len(remaining_matches)} matches, {len(remaining_non_matches)} non-matches")

val_matches = remaining_matches.sample(n=500, random_state=0)
val_non_matches = remaining_non_matches.sample(n=500, random_state=0)
# マッチ・非マッチを結合してシャッフルし、順序バイアスを除去
val_df = pd.concat([val_matches, val_non_matches]).sample(frac=1, random_state=0).reset_index(drop=True)
print(f"Val: {len(val_df)} pairs, {val_df['label'].sum()} matches (50%)")

# val/testの重複がないことを確認
test_pairs = set(zip(test_df["idAbt"], test_df["idBuy"]))
val_pairs = set(zip(val_df["idAbt"], val_df["idBuy"]))
assert len(test_pairs & val_pairs) == 0, "val/test overlap detected!"
print("val/test overlap check: OK (0 overlaps)")

# ===== CSV保存 =====

# abt_buy_candidates.csvと同形式（idAbt, idBuy, label）で出力
# index=False なので pandas インデックスは出力不要
val_df[["idAbt", "idBuy", "label"]].to_csv("dataset/Abt-Buy/sampled_em_val_data.csv", index=False)
test_df[["idAbt", "idBuy", "label"]].to_csv("dataset/Abt-Buy/sampled_em_test_data.csv", index=False)

print("Saved sampled_em_val_data.csv and sampled_em_test_data.csv")
