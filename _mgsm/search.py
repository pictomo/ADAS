# MGSMタスクにおけるエージェントアーキテクチャの自動探索・評価を行うメインモジュール
# LLMを用いて新しいエージェント設計を提案・評価し、進化的に最適なエージェントを探索する

import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from mgsm_prompt import get_init_archive, get_prompt, get_reflexion_prompt

# OpenAI APIクライアントの初期化
client = openai.OpenAI()

from utils import get_all_examples, random_id, bootstrap_confidence_interval, score_mgsm

# エージェント間でやり取りされる情報を保持する名前付きタプル
# name: フィールド名, author: 生成者, content: 内容, iteration_idx: イテレーション番号
Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])

# LLMの出力フォーマットを指定するテンプレート（JSON形式での応答を強制）
FORMAT_INST = (
    lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
)
# LLMの役割記述テンプレート
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

# デバッグ出力フラグ
PRINT_LLM_DEBUG = False
# 探索モードフラグ（True=検証データで探索、False=テストデータで評価）
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    """GPTモデルからJSON形式のレスポンスを取得する。

    レートリミット時には指数バックオフで自動リトライする。
    単一メッセージの問い合わせに使用する。

    Args:
        msg (str): ユーザーメッセージ。
        model (str): 使用するGPTモデル名。
        system_message (str): システムメッセージ。
        temperature (float): サンプリング温度。

    Returns:
        dict: パース済みのJSONディクショナリ。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=4096,
        stop=None,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(msg_list, model, temperature=0.8):
    """GPTモデルからリフレクション用のJSONレスポンスを取得する。

    複数メッセージの会話履歴（msg_list）を渡して、リフレクションや
    デバッグのための多ターン対話に使用する。
    レートリミット時には指数バックオフで自動リトライする。

    Args:
        msg_list (list[dict]): メッセージ履歴のリスト（role/contentの辞書）。
        model (str): 使用するGPTモデル名。
        temperature (float): サンプリング温度。

    Returns:
        dict: パース済みのJSONディクショナリ。
    """
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=4096,
        stop=None,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase:
    """LLMエージェントの基底クラス。

    プロンプトの構築、GPTへの問い合わせ、応答のパースを担当する。
    各エージェントは役割・温度・出力フィールドをカスタマイズできる。

    Attributes:
    """

    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role="helpful assistant",
        model="gpt-3.5-turbo-0125",
        temperature=0.5,
    ) -> None:
        """LLMエージェントを初期化する。

        Args:
            output_fields (list): LLMに出力させるフィールド名のリスト（例: ['thinking', 'answer']）。
            agent_name (str): エージェントの名前。
            role (str): LLMに与える役割の説明。
            model (str): 使用するGPTモデル名。
            temperature (float): サンプリング温度。
        """
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        """LLMに送るプロンプトを構築する。

        入力情報（Infoオブジェクトのリスト）と指示文から、
        システムプロンプトとユーザープロンプトを生成する。
        answerフィールドには整数のみを返す旨の指示が付加される。

        Args:
            input_infos (list): 入力情報のInfoオブジェクトのリスト。
            instruction (str): タスクの指示文。

        Returns:
            tuple: (システムプロンプト, ユーザープロンプト)
        """
        # システムプロンプトの構築（役割 + 出力フォーマット指定）
        # construct system prompt
        output_fields_and_description = {
            key: (
                f"Your {key}."
                if not "answer" in key
                else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer."
            )
            for key in self.output_fields
        }
        system_prompt = (
            ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)
        )

        # 入力情報をMarkdown形式のテキストに変換
        # construct input infos text
        input_infos_text = ""
        for input_info in input_infos:
            if isinstance(input_info, Info):
                field_name, author, content, iteration_idx = input_info
            else:
                continue
            if author == self.__repr__():
                author += " (yourself)"
            if field_name == "task":
                input_infos_text += f"# Your Task:\n{content}\n\n"
            elif iteration_idx != -1:
                input_infos_text += (
                    f"### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n"
                )
            else:
                input_infos_text += f"### {field_name} by {author}:\n{content}\n\n"

        # 入力情報テキストと指示文を結合して最終プロンプトを生成
        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        """入力情報と指示文を基にLLMに問い合わせ、結果をInfoリストとして返す。

        プロンプトを構築してGPTに問い合わせ、JSONレスポンスをパースして
        Infoオブジェクトのリストに変換する。エラー時は不足フィールドの補完を試みる。

        Args:
            input_infos (list): 入力情報のInfoオブジェクトのリスト。
            instruction (str): タスクの指示文。
            iteration_idx (int): イテレーション番号（-1は番号なし）。

        Returns:
            list[Info]: 出力情報のInfoオブジェクトのリスト。
        """
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(
                prompt, self.model, system_prompt, self.temperature
            )
            assert len(response_json) == len(
                self.output_fields
            ), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError(
                    "The context is too long. Please try to design the agent to have shorter context."
                )
            # 不足フィールドを空文字で補完し、余分なフィールドを削除する
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(
                    self.output_fields
                ):
                    response_json[key] = ""
            for key in copy.deepcopy(list(response_json.keys())):
                if (
                    len(response_json) > len(self.output_fields)
                    and not key in self.output_fields
                ):
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem:
    """MGSMタスクを解くエージェントシステム。

    forward()メソッドは探索中に動的に差し替えられる。
    """

    def __init__(self) -> None:
        pass


def search(args):
    """エージェントアーキテクチャの探索を実行する。

    初期アーカイブの評価後、LLMを使って新しいエージェントの提案・リフレクション・
    評価を繰り返し、進化的にアーカイブを拡張していく。
    結果は各世代ごとにJSONファイルに保存される。

    Args:
        args: コマンドライン引数（モデル名、世代数、保存先など）。
    """
    # 既存のアーカイブがあれば読み込み、なければ初期アーカイブで開始
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]["generation"], int):
            start = archive[-1]["generation"]
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if "fitness" in solution:
            continue

        solution["generation"] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        # ブートストラップ信頼区間を算出して適応度として保存
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution["fitness"] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(archive, json_file, indent=4)

    # 各世代で新しいエージェントを提案・リフレクション・評価するループ
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        # メタプロンプトを構築して新しいエージェントを提案させる
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            # 2回のリフレクションで設計を洗練させる
            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(
                archive[-1] if n > 0 else None
            )
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1
            continue

        # 提案されたエージェントの評価（失敗時はデバッグを試みる）
        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append(
                    {
                        "role": "user",
                        "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'",
                    }
                )
                try:
                    next_solution = get_json_response_from_gpt_reflect(
                        msg_list, args.model
                    )
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            n -= 1
            continue

        # 適応度を算出し、不要なフィールドを除去してアーカイブに追加
        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution["fitness"] = fitness_str
        next_solution["generation"] = n + 1

        if "debug_thought" in next_solution:
            del next_solution["debug_thought"]
        if "reflection" in next_solution:
            del next_solution["reflection"]
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    """探索済みアーカイブの全エージェントをテストデータで評価する。

    探索フェーズで見つけた各エージェントをテストデータに対して評価し、
    結果を別のJSONファイルに保存する。

    Args:
        args: コマンドライン引数。
    """
    # 探索アーカイブと評価アーカイブのファイルパスを設定
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = (
        str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(
            ".json"
        )
        + "_evaluate.json"
    )
    with open(file_path, "r") as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as json_file:
            eval_archive = json.load(json_file)

    # アーカイブ内の各エージェントを順番にテスト評価
    current_idx = 0
    while current_idx < len(archive):
        with open(file_path, "r") as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol["test_fitness"] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, "w") as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    """エージェントのforward関数を動的に定義し、MGSMデータセットで評価する。

    文字列として渡されたforward関数をexecで動的に定義し、
    AgentSystemにセットして、マルチスレッドでMGSMタスク群を評価する。
    探索モードでは検証データ、評価モードではテストデータを使用する。

    Args:
        args: コマンドライン引数（データサイズ、シャッフルシード、ワーカー数など）。
        forward_str (str): forward()関数のソースコード文字列。

    Returns:
        list[float]: 各問題の正誤スコア（正解=1, 不正解=0）のリスト。
    """
    # forward関数を動的に定義してAgentSystemにセット
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # 全例題を取得してシード固定でシャッフルし、探索/テストモードに応じてスライス
    # set seed 0 for valid set
    examples = get_all_examples()
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[: args.valid_size] * args.n_repreat
    else:
        examples = (
            examples[args.valid_size : args.valid_size + args.test_size]
            * args.n_repreat
        )

    questions = [example["inputs"] for example in examples]
    answers = [example["targets"] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    # 各問題をInfoオブジェクトに変換してタスクキューを作成
    task_queue = []
    for q in questions:
        taskInfo = Info("task", "User", q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    # マルチスレッドで全タスクを並列評価
    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue))
        )

    # 各結果を正解と比較してスコアを算出
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            correct_answer = answers[q_idx]
            correct = score_mgsm(correct_answer, extracted_answer)
        except Exception as e:
            acc_list.append(0)
            continue

        acc_list.append(1 if correct else 0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=800)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--n_repreat", type=int, default=1)
    parser.add_argument("--multiprocessing", action="store_true", default=True)
    parser.add_argument("--max_workers", type=int, default=48)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--expr_name", type=str, default="mgsm_gpt3.5_results")
    parser.add_argument("--n_generation", type=int, default=30)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=["gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125", "gpt-4o-2024-05-13"],
    )

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
