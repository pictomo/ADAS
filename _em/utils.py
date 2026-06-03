# EMタスクの評価・データ取得に関するユーティリティモジュール

import random
import string
from functools import lru_cache

import numpy as np
import pandas as pd


def random_id(length=4):
    """ランダムな英数字IDを生成する。

    大文字・小文字のアルファベットと数字を組み合わせて、指定された長さのランダムIDを返す。
    エージェントインスタンスの一意識別子として使用される。

    Args:
        length (int): 生成するIDの文字数。デフォルトは4。

    Returns:
        str: ランダムに生成された英数字の文字列。
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choices(characters, k=length))


def _format_value(v) -> str:
    """値を文字列に変換する。NaN/Noneは空文字として扱う。"""
    if pd.isna(v):
        return ""
    return str(v).strip()


def _format_entity_pair(idAbt: int, idBuy: int, abt_dict: dict, buy_dict: dict) -> str:
    """AbtとBuyのエンティティペアをLLMへの入力テキストに整形する。

    Args:
        idAbt (int): AbtエンティティのID。
        idBuy (int): BuyエンティティのID。
        abt_dict (dict): AbtエンティティIDから属性辞書へのマッピング。
        buy_dict (dict): BuyエンティティIDから属性辞書へのマッピング。

    Returns:
        str: "Entity A: ..." および "Entity B: ..." 形式の整形済みテキスト。
    """
    a = abt_dict.get(idAbt, {})
    b = buy_dict.get(idBuy, {})
    abt_text = (
        f"Entity A:\n"
        f"  Name: {_format_value(a.get('name', ''))}\n"
        f"  Description: {_format_value(a.get('description', ''))}\n"
        f"  Price: {_format_value(a.get('price', ''))}"
    )
    buy_text = (
        f"Entity B:\n"
        f"  Name: {_format_value(b.get('name', ''))}\n"
        f"  Description: {_format_value(b.get('description', ''))}\n"
        f"  Manufacturer: {_format_value(b.get('manufacturer', ''))}\n"
        f"  Price: {_format_value(b.get('price', ''))}"
    )
    return abt_text + "\n\n" + buy_text


@lru_cache(maxsize=2)
def get_all_examples(split: str = "val") -> list[dict]:
    """サンプリング済みCSVとエンティティCSVからEM例題リストを構築する。

    サンプリング済みのペアCSV（idAbt, idBuy, label）を読み込み、
    Abt.csv・Buy.csv と結合してLLMに渡す入力テキストを生成する。
    lru_cache により val/test それぞれ初回のみCSVを読み込み、以降はキャッシュを返す。
    リポジトリルートから実行されることを前提としてパスを解決する。

    Args:
        split (str): "val"（検証セット）または "test"（テストセット）。

    Returns:
        list[dict]: 各要素が {"inputs": str, "targets": bool, "idAbt": int, "idBuy": int} の辞書リスト。

    Raises:
        ValueError: splitが "val" または "test" 以外の場合。
    """
    if split == "val":
        pairs_path = "dataset/Abt-Buy/sampled_em_val_data.csv"
    elif split == "test":
        pairs_path = "dataset/Abt-Buy/sampled_em_test_data.csv"
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'val' or 'test'.")

    # サンプリング済みペアと各エンティティのメタデータを読み込む
    pairs = pd.read_csv(pairs_path)
    abt = pd.read_csv("dataset/Abt-Buy/Abt.csv")
    buy = pd.read_csv("dataset/Abt-Buy/Buy.csv")

    # 高速参照のためIDキーの辞書に変換（pandas Seriesの副作用を避けるためto_dictを使用）
    abt_dict = {row["id"]: row.to_dict() for _, row in abt.iterrows()}
    buy_dict = {row["id"]: row.to_dict() for _, row in buy.iterrows()}

    examples = []
    for _, row in pairs.iterrows():
        idAbt, idBuy = int(row["idAbt"]), int(row["idBuy"])
        examples.append(
            {
                "inputs": _format_entity_pair(idAbt, idBuy, abt_dict, buy_dict),
                "targets": bool(row["label"]),
                "idAbt": idAbt,
                "idBuy": idBuy,
            }
        )
    return examples


def parse_em_prediction(prediction) -> bool:
    """LLMの出力をbool（マッチ/非マッチ）に変換する。

    JSON boolean（Python bool型）や文字列 "true"/"false" を解釈する。
    判定不能な場合は False（非マッチ）をデフォルトとする。

    Args:
        prediction: Info.content から取り出した値（bool, str, int, None など）。

    Returns:
        bool: True=マッチ、False=非マッチ。
    """
    # JSON boolean として返ってきた場合はそのまま返す
    if isinstance(prediction, bool):
        return prediction
    if isinstance(prediction, int):
        return bool(prediction)
    if isinstance(prediction, str):
        s = prediction.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        # 文字列内に "true"/"false" が混在する場合のファジーマッチ
        has_true = "true" in s
        has_false = "false" in s
        if has_true and not has_false:
            return True
        if has_false and not has_true:
            return False
    # 判定不能な場合はデフォルトの非マッチとして扱う
    return False


def compute_f1(label_pairs: list[tuple]) -> float:
    """(y_true, y_pred) ペアのリストからF1スコアを計算する。

    エンティティマッチングにおいてマッチ（True）を正例とする。
    precision と recall の調和平均として F1 を算出する。

    Args:
        label_pairs (list[tuple]): [(y_true: bool, y_pred: bool), ...] のリスト。

    Returns:
        float: F1スコア [0.0, 1.0]。正例予測も正例ラベルも0件のときは 0.0 を返す。
    """
    tp = sum(1 for y_true, y_pred in label_pairs if y_true and y_pred)
    fp = sum(1 for y_true, y_pred in label_pairs if not y_true and y_pred)
    fn = sum(1 for y_true, y_pred in label_pairs if y_true and not y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bootstrap_confidence_interval(
    label_pairs: list[tuple],
    num_bootstrap_samples: int = 100000,
    confidence_level: float = 0.95,
) -> str:
    """ブートストラップ法によりF1スコアの信頼区間を算出する。

    各リサンプルでF1を再計算することで、線形でないF1の特性を正しく扱う。
    MGSMのaccuracy版と同じ出力フォーマット（パーセント文字列）を維持する。
    numpy の vectorized インデックス生成により高速化している。

    Calculate the bootstrap confidence interval for the F1 score of entity matching.
    Also returns the median of the bootstrap F1 scores.

    Args:
        label_pairs (list[tuple]): [(y_true: bool, y_pred: bool), ...] のリスト。
        num_bootstrap_samples (int): ブートストラップサンプル数。
        confidence_level (float): 信頼水準（例: 0.95 は 95% 信頼区間）。

    Returns:
        str: "95% Bootstrap Confidence Interval: (X.X%, Y.Y%), Median: Z.Z%" 形式の文字列。
    """
    if not label_pairs:
        return "95% Bootstrap Confidence Interval: (0.0%, 0.0%), Median: 0.0%"

    n = len(label_pairs)

    # numpy配列に変換してベクトル演算を可能にする
    y_true_arr = np.array([int(y_true) for y_true, _ in label_pairs])
    y_pred_arr = np.array([int(y_pred) for _, y_pred in label_pairs])

    # 全リサンプル分のインデックスを一括生成（ループ内random.choicesより高速）
    indices = np.random.randint(0, n, size=(num_bootstrap_samples, n))

    def _f1_from_arrays(yt, yp):
        """bool配列からF1スコアを計算するヘルパー関数。"""
        tp = np.sum(yt & yp)
        fp = np.sum(~yt & yp)
        fn = np.sum(yt & ~yp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # 各リサンプルについてF1スコアを計算
    bootstrap_f1s = np.array([
        _f1_from_arrays(
            y_true_arr[indices[i]].astype(bool),
            y_pred_arr[indices[i]].astype(bool),
        )
        for i in range(num_bootstrap_samples)
    ])

    # 信頼区間の下限・上限に対応するパーセンタイルを計算
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_f1s, lower_percentile * 100) * 100
    ci_upper = np.percentile(bootstrap_f1s, upper_percentile * 100) * 100

    # ブートストラップF1値群の中央値を算出
    median = np.median(bootstrap_f1s) * 100

    # パーセント表記に変換して整形済み文字列として返す
    return f"95% Bootstrap Confidence Interval: ({ci_lower:.1f}%, {ci_upper:.1f}%), Median: {median:.1f}%"
