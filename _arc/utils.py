# modified from https://github.com/jennyzzt/LLM_debate_on_ARC
# prompt also inspired by https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/master/arc_solve/prompting.py

# ARC (Abstraction and Reasoning Corpus) タスクの評価・データ整形に関するユーティリティモジュール

import concurrent.futures
import random
import string

import numpy as np

# ARCタスクの概要説明文（LLMへのプロンプトとして使用される）
TASK_OVERVIEW = """You will be given some number of paired example inputs and outputs grids. The outputs were produced by applying a transformation rule to the input grids. In addition to the paired example inputs and outputs, there is also one test input without a known output.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color. 0 is black.
Your task is to determine the transformation rule from examples and find out the answer, involving determining the size of the output grid for the test and correctly filling each cell of the grid with the appropriate color or number.

The transformation only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the grid visually, and try to find the pattern.
"""


def random_id(length=4):
    """ランダムな英数字IDを生成する。

    大文字・小文字のアルファベットと数字を組み合わせて、指定された長さのランダムIDを返す。
    タスクやセッションの一意識別子として使用される。

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


def file_to_string(filepath):
    """ファイルの内容を文字列として読み込む。

    指定されたファイルパスからテキストを読み込み、前後の空白を除去して返す。

    Args:
        filepath (str): 読み込むファイルのパス。

    Returns:
        str: ファイルの内容（前後の空白除去済み）。
    """
    with open(filepath, "r") as f:
        data = f.read().strip()
    return data


def list_to_string(list_2d):
    """2次元リスト（グリッド）を文字列表現に変換する。

    ARCタスクのグリッドデータをプロンプトに埋め込むための文字列に変換する。
    例: [[1,2],[3,4]] → "[[1,2],[3,4]]"

    Args:
        list_2d (list[list[int]]): 2次元のグリッドデータ。

    Returns:
        str: グリッドの文字列表現。
    """
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"


def format_arc_data(arc_data, direct=False):
    """ARCタスクデータをLLMプロンプト用の文字列に整形する。

    訓練用の入出力例とテスト問題をMarkdown形式の文字列にフォーマットし、
    LLMに提示するためのプロンプトを構築する。

    Args:
        arc_data (dict): ARCタスクデータ。'train'キーに訓練例のリスト、
            'test'キーにテストケースのリストを含む辞書。
        direct (bool): 直接出力モードのフラグ（現在未使用）。

    Returns:
        tuple: (タスク全体のプロンプト文字列, 訓練データのリスト, テスト入力グリッド)
    """
    task_str = TASK_OVERVIEW

    # 訓練用の入出力例をMarkdown形式で構築
    task_demo_str = ""
    # Get task demo string
    task_demo_str += "## Examples:\n\n"
    for i, demo in enumerate(arc_data["train"]):
        task_demo_str += f"### Example {i}:\n"
        task_demo_str += f'input = {list_to_string(demo["input"])}\n'
        task_demo_str += f'output = {list_to_string(demo["output"])}\n\n'

    # テスト問題の入力と指示文を構築
    # Get task test string
    task_test_str = ""
    for testcase in arc_data["test"]:
        task_test_str += "## Test Problem:\n"
        task_test_str += f'Given input:\n {list_to_string(testcase["input"])}\n\n'
        task_test_str += f"Analyze the transformation rules based on the provided Examples and determine what the output should be for the Test Problem."

    # 概要・例題・テスト問題を結合して最終プロンプトを生成
    task_str += task_demo_str + task_test_str

    return task_str, arc_data["train"], arc_data["test"][0]["input"]


def get_percentage_match(arr1, arr2):
    """2つのグリッド間のセル一致率を計算する（ソフト評価用）。

    正解グリッド(arr1)と予測グリッド(arr2)を要素ごとに比較し、
    一致するセルの割合を返す。サイズが異なる場合も部分的に比較を行う。

    Args:
        arr1 (list[list[int]]): 正解のグリッド。
        arr2 (list[list[int]]): 予測（生成された）グリッド。

    Returns:
        float: 一致率（0.0〜1.0）。arr2が空の場合は0を返す。
    """
    # arr1 is solution
    if not arr2:
        return 0
    # 各セルを走査し、一致するセル数をカウント
    score = 0
    for i, xs in enumerate(arr1):
        try:
            for j, x in enumerate(xs):
                try:
                    if len(arr2) > i and len(arr2[i]) > j and arr2[i][j] == x:
                        score += 1
                except:
                    pass
        except:
            pass
    # 正解グリッドの総セル数で割って一致率を算出
    score = score / (len(arr1) * len(arr1[0]))
    return score


def eval_algo(solve_fn, arc_data, soft_eval=False):
    """解答アルゴリズム（関数）の正答率を評価する。

    与えられた解答関数を各テストケースに適用し、正解との一致度を評価する。
    タイムアウト（30秒）やエラー発生時にも安全に処理を続行する。

    Args:
        solve_fn (callable): 入力グリッドを受け取り出力グリッドを返す解答関数。
        arc_data (dict): ARCタスクデータ（'test'キーにテストケースのリストを含む）。
        soft_eval (bool): Trueの場合、セル単位の部分一致率で評価。
            Falseの場合、完全一致のみを正解とする。

    Returns:
        float: 全テストケースの平均スコア。
    """
    # Calculate percentage of test cases done correctly
    testcases = arc_data["test"]
    scores = []
    for testcase in testcases:
        input = testcase["input"]
        output = testcase["output"]
        gen_output = None
        # 30秒のタイムアウト付きで解答関数を実行
        # Run solve_fn with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(solve_fn, input)
                try:
                    gen_output = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    future.cancel()
            except:  # if the function does not work
                continue
        # 正解と比較してスコアを算出（ソフト評価 or 完全一致）
        # Check if correct output
        if soft_eval:
            score = get_percentage_match(output, gen_output)
        else:
            score = 1 if output == gen_output else 0
        scores.append(score)
    return np.mean(scores)


def eval_solution(output, arc_data, soft_eval=False):
    """単一の出力グリッドを正解と比較して評価する。

    eval_algoと異なり、解答関数ではなく既に生成済みの出力グリッドを
    直接評価する。最初のテストケースの正解と比較する。

    Args:
        output (list[list[int]] | None): 評価対象の出力グリッド。
        arc_data (dict): ARCタスクデータ（正解を含む）。
        soft_eval (bool): Trueの場合、セル単位の部分一致率で評価。
            Falseの場合、完全一致のみを正解とする。

    Returns:
        float: スコア（完全一致なら0または1、ソフト評価なら0.0〜1.0）。
            outputがNone/空の場合は0を返す。
    """
    if not output:
        return 0

    # 最初のテストケースの正解を取得して比較
    solution = arc_data["test"][0]["output"]
    if soft_eval:
        score = get_percentage_match(solution, output)
    else:
        score = 1 if output == solution else 0
    return score


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
