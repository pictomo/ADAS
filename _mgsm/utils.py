# MGSM (Multilingual Grade School Math) タスクの評価・データ取得に関するユーティリティモジュール

import random
import string
from collections import namedtuple

import numpy as np

# エージェント間でやり取りされる情報を保持する名前付きタプル（現在このモジュールでは未使用）
Example = namedtuple(
    "Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]
)

# 言語コードから数学問題への指示文へのマッピング
# {input} には実際の問題文が挿入される
LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem.

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন।

{input}""",
    "de": """Löse dieses Mathematikproblem.

{input}""",
    "es": """Resuelve este problema matemático.

{input}""",
    "fr": """Résolvez ce problème de mathématiques.

{input}""",
    "ja": """この数学の問題を解いてください。

{input}""",
    "ru": """Решите эту математическую задачу.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu.

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้

{input}""",
    "zh": """解决这个数学问题。

{input}""",
}

# 言語コードからMGSMデータセットのファイルパスを生成するラムダ
LANG_TO_FPATH = lambda lang: f"dataset/mgsm/mgsm_{lang}.tsv"

# MGSMがサポートする全11言語のコードリスト
ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def score_mgsm(target: str, prediction: str) -> bool:
    """MGSMタスクの予測値と正解値を比較して正誤を判定する。

    末尾の不要なゼロやカンマを除去して正規化した上で文字列比較を行う。
    小数点以下が全てゼロの場合（例: "3.00"）は整数として扱う。

    Args:
        target (str): 正解の数値文字列。
        prediction (str): LLMが生成した予測の数値文字列。

    Returns:
        bool: 正規化後の文字列が一致すればTrue、それ以外はFalse。
    """
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def get_lang_examples(lang: str) -> list[dict[str, str]]:
    """指定言語のMGSMデータセットを読み込んで例題リストを返す。

    TSV形式のファイル（問題文\t正解）を各行パースし、言語別の指示文を付加して
    inputs/targets/lang のキーを持つ辞書のリストを生成する。

    Args:
        lang (str): 言語コード（例: "en", "ja", "zh"）。

    Returns:
        list[dict[str, str]]: 各要素が {"inputs": 指示文付き問題文, "targets": 正解文字列, "lang": 言語コード} の辞書リスト。

    Raises:
        ValueError: 正解に小数点が含まれる場合（MGSMは整数のみを想定）。
    """
    fpath = LANG_TO_FPATH(lang)
    examples = []
    with open(fpath, mode="r", encoding="utf-8") as f:
        for line in f:
            inputs, targets = line.strip().split("\t")
            if "." in targets:
                raise ValueError(f"targets {targets} contains a decimal point.")
            # targets = int(targets.replace(",", ""))
            examples.append(
                {
                    "inputs": LANG_TO_INSTRUCTIONS[lang].format(input=inputs),
                    "targets": targets,
                    "lang": lang,
                }
            )
    return examples


def get_all_examples() -> list[dict[str, str]]:
    """全11言語のMGSMデータセットを結合して返す。

    ALL_LANGUAGES に列挙された全言語の例題を順に読み込み、
    一つのリストとして結合する。

    Returns:
        list[dict[str, str]]: 全言語の例題をまとめたリスト。
    """
    examples = []
    for lang in ALL_LANGUAGES:
        # if lang != "en":
        #     continue
        examples += get_lang_examples(lang)
    return examples


def random_id(length=4):
    """ランダムな英数字IDを生成する。

    大文字・小文字のアルファベットと数字を組み合わせて、指定された長さのランダムIDを返す。
    エージェントインスタンスの一意識別子として使用される。

    Args:
        length (int): 生成するIDの文字数。デフォルトは4。

    Returns:
        str: ランダムに生成された英数字の文字列。
    """
    characters = (
        string.ascii_letters + string.digits
    )  # includes both upper/lower case letters and numbers
    random_id = "".join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(
    data, num_bootstrap_samples=100000, confidence_level=0.95
):
    """ブートストラップ法により精度データの信頼区間を算出する。

    1次元の精度データから復元抽出を繰り返し、平均値の分布を推定することで
    信頼区間と中央値を計算する。結果はパーセント表記の文字列として返される。

    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.

    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # 信頼区間の下限・上限に対応するパーセンタイルを計算
    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # ブートストラップ平均値群の中央値を算出
    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # パーセント表記に変換して整形済み文字列として返す
    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
