# エンティティマッチングタスクにおけるエージェントアーキテクチャの自動探索・評価を行うメインモジュール
# LLMを用いて新しいエージェント設計を提案・評価し、進化的に最適なエージェントを探索する

import argparse
import copy
import json
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from em_prompt import get_init_archive, get_prompt, get_reflexion_prompt

# OpenAI APIクライアントの初期化
client = openai.OpenAI()

from utils import (
    get_all_examples,
    random_id,
    bootstrap_confidence_interval,
    parse_em_prediction,
    compute_f1,
)

# エージェント間でやり取りされる情報を保持する名前付きタプル
# name: フィールド名, author: 生成者, content: 内容, iteration_idx: イテレーション番号
Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])

# LLMの出力フォーマットを指定するテンプレート（JSON形式での応答を強制）
FORMAT_INST = (
    lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
)
# LLMの役割記述テンプレート
ROLE_DESC = lambda role: f"You are a {role}."
# デバッグ出力フラグ（--debug フラグで ON になる）
PRINT_LLM_DEBUG = False
# 探索モードフラグ（True=検証データで探索、False=テストデータで評価）
SEARCHING_MODE = True
# エージェント評価用モデル（__main__で args.eval_model から上書きされる）
EVAL_MODEL = "gpt-5-nano"
# エージェント評価用推論量（--eval_reasoning_effort から上書きされる）
EVAL_REASONING_EFFORT = "minimal"
# メタLLM用推論量（--meta_reasoning_effort から上書きされる）
META_REASONING_EFFORT = "none"


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
def get_json_response_from_gpt(msg, model, system_message):
    """GPTモデルからJSON形式のレスポンスを取得する。

    レートリミット時には指数バックオフで自動リトライする。
    単一メッセージの問い合わせに使用する。
    EVAL_REASONING_EFFORT でエージェント評価の推論量を制御する。

    Args:
        msg (str): ユーザーメッセージ。
        model (str): 使用するGPTモデル名。
        system_message (str): システムメッセージ。

    Returns:
        dict: パース済みのJSONディクショナリ。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        reasoning_effort=EVAL_REASONING_EFFORT,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    assert content is not None
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
def get_json_response_from_gpt_reflect(msg_list, model):
    """GPTモデルからリフレクション用のJSONレスポンスを取得する。

    複数メッセージの会話履歴（msg_list）を渡して、リフレクションや
    デバッグのための多ターン対話に使用する。
    レートリミット時には指数バックオフで自動リトライする。
    META_REASONING_EFFORT でメタLLMの推論量を制御する。

    Args:
        msg_list (list[dict]): メッセージ履歴のリスト（role/contentの辞書）。
        model (str): 使用するGPTモデル名。

    Returns:
        dict: パース済みのJSONディクショナリ。
    """
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        reasoning_effort=META_REASONING_EFFORT,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    assert content is not None
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


class LLMAgentBase:
    """LLMエージェントの基底クラス。

    プロンプトの構築、GPTへの問い合わせ、応答のパースを担当する。
    各エージェントは役割・温度・出力フィールドをカスタマイズできる。

    Attributes:
        output_fields (list): LLMに出力させるフィールド名のリスト（例: ['thinking', 'answer']）。
        agent_name (str): エージェントの名前。
        role (str): LLMに与える役割の説明。
        model (str): 使用するGPTモデル名。
        id (str): エージェントインスタンスの一意識別子。
    """

    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role="helpful assistant",
        model=None,
    ) -> None:
        """LLMエージェントを初期化する。

        Args:
            output_fields (list): LLMに出力させるフィールド名のリスト。
            agent_name (str): エージェントの名前。
            role (str): LLMに与える役割の説明。
            model (str | None): 使用するGPTモデル名。None の場合は EVAL_MODEL を使用。
        """
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model if model is not None else EVAL_MODEL
        # 各インスタンスに一意のIDを付与
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> tuple[str, str]:
        """LLMに送るプロンプトを構築する。

        入力情報（Infoオブジェクトのリスト）と指示文から、
        システムプロンプトとユーザープロンプトを生成する。
        answerフィールドにはtrue/falseのみを返す旨の指示が付加される。

        Args:
            input_infos (list): 入力情報のInfoオブジェクトのリスト。
            instruction (str): タスクの指示文。

        Returns:
            tuple: (システムプロンプト, ユーザープロンプト)
        """
        # システムプロンプトの構築（役割 + 出力フォーマット指定）
        output_fields_and_description = {
            key: (
                f"Your {key}."
                if "answer" not in key
                else f"Your {key}. Return ONLY true (match) or false (non-match). DO NOT return anything other than true or false."
            )
            for key in self.output_fields
        }
        system_prompt = (
            ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)
        )

        # 入力情報をMarkdown形式のテキストに変換
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

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
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
        response_json: dict = {}
        try:
            response_json = get_json_response_from_gpt(
                prompt, self.model, system_prompt
            )
            assert len(response_json) == len(
                self.output_fields
            ), "not returning enough fields"
        except Exception as e:
            if PRINT_LLM_DEBUG:
                print(f"[LLMAgent DEBUG] exception in query: {type(e).__name__}: {e}")
                print(f"[LLMAgent DEBUG] response_json at exception: {response_json}")
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError(
                    "The context is too long. Please try to design the agent to have shorter context."
                )
            # 不足フィールドを空文字で補完し、余分なフィールドを削除する
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
    """エンティティマッチングタスクを解くエージェントシステム。

    forward()メソッドは探索中に動的に差し替えられる。
    """

    def __init__(self) -> None:
        pass

    def forward(self, _taskInfo: Info) -> Info | str:
        raise NotImplementedError


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
        # 評価前保存で中断された未完成エントリ（fitness無し生成エージェント）を再評価して完了させる
        if (
            archive
            and isinstance(archive[-1].get("generation"), int)
            and "fitness" not in archive[-1]
        ):
            incomplete = archive[-1]
            print(
                f"Resuming evaluation of incomplete entry for generation {incomplete['generation']}"
            )
            label_pairs = []
            try:
                label_pairs = evaluate_forward_fn(args, incomplete["code"])
            except Exception as e:
                print(f"Re-evaluation failed: {e}")
            if label_pairs and compute_f1(label_pairs) >= 0.01:
                fitness_str = bootstrap_confidence_interval(label_pairs)
                archive[-1]["fitness"] = fitness_str
                for key in ("debug_thought", "reflection"):
                    archive[-1].pop(key, None)
            else:
                archive.pop()
            with open(file_path, "w") as json_file:
                json.dump(archive, json_file, indent=4)
        if archive and isinstance(archive[-1].get("generation"), int):
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

        # 評価前にアーカイブを書き出し（クラッシュ時にどのエージェントを評価中か確認可能）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(archive, json_file, indent=4)

        try:
            label_pairs = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        # ブートストラップ信頼区間を算出して適応度として保存
        fitness_str = bootstrap_confidence_interval(label_pairs)
        solution["fitness"] = fitness_str

        # fitness追加後に再度書き出し
        with open(file_path, "w") as json_file:
            json.dump(archive, json_file, indent=4)

    # 各世代で新しいエージェントを提案・リフレクション・評価するループ
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        # メタプロンプトを構築して新しいエージェントを提案させる
        system_prompt, prompt = get_prompt(archive, EVAL_MODEL, EVAL_REASONING_EFFORT)
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
            continue

        # 評価前にアーカイブに追加して書き出し（クラッシュ時にどのコードを評価中か確認可能）
        next_solution["generation"] = n + 1
        archive.append(next_solution)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(archive, json_file, indent=4)

        # 提案されたエージェントの評価（失敗時はデバッグを試みる）
        label_pairs = []
        for _ in range(args.debug_max):
            try:
                label_pairs = evaluate_forward_fn(args, archive[-1]["code"])
                if compute_f1(label_pairs) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 F1")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(archive[-1])})
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
                    next_solution["generation"] = n + 1
                    archive[-1] = next_solution
                    with open(file_path, "w") as json_file:
                        json.dump(archive, json_file, indent=4)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue

        if not label_pairs:
            # 評価が全て失敗した場合はアーカイブから除去して保存
            archive.pop()
            with open(file_path, "w") as json_file:
                json.dump(archive, json_file, indent=4)
            continue

        # 適応度を算出し、不要なフィールドを除去して最終保存
        fitness_str = bootstrap_confidence_interval(label_pairs)
        archive[-1]["fitness"] = fitness_str
        if "debug_thought" in archive[-1]:
            del archive[-1]["debug_thought"]
        if "reflection" in archive[-1]:
            del archive[-1]["reflection"]

        # save results
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
        os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json").removesuffix(
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
            label_pairs = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(label_pairs)
        sol["test_fitness"] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, "w") as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    """エージェントのforward関数を動的に定義し、EMデータセットで評価する。

    文字列として渡されたforward関数をexecで動的に定義し、
    AgentSystemにセットして、マルチスレッドでEMタスク群を評価する。
    探索モードでは検証データ、評価モードではテストデータを使用する。

    Args:
        args: コマンドライン引数（データサイズ、ワーカー数など）。
        forward_str (str): forward()関数のソースコード文字列。

    Returns:
        list[tuple]: 各問題の (y_true: bool, y_pred: bool) ペアのリスト。
    """
    # forward関数を動的に定義してAgentSystemにセット
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # 探索/テストモードに応じてデータを読み込む（前処理済みCSVから）
    examples = get_all_examples("val" if SEARCHING_MODE else "test")
    examples = examples * args.n_repeat

    questions = [example["inputs"] for example in examples]
    answers = [example["targets"] for example in examples]  # list[bool]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    # 各問題をInfoオブジェクトに変換してタスクキューを作成
    task_queue = []
    for q in questions:
        taskInfo = Info("task", "User", q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    # マルチスレッドで全タスクを並列評価
    label_pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue))
        )

    # 各結果を正解ラベルと比較して (y_true, y_pred) ペアを構築
    if PRINT_LLM_DEBUG:
        for i, r in enumerate(results[:3]):
            raw = r.content if isinstance(r, Info) else r
            print(f"[DEBUG] sample[{i}] raw={raw!r}  true_label={answers[i]}")

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            predicted = parse_em_prediction(extracted_answer)
            correct_label = answers[q_idx]
        except Exception:
            # エラー時はデフォルトの非マッチとして記録
            label_pairs.append((answers[q_idx], False))
            continue
        label_pairs.append((correct_label, predicted))

    print(f"F1 (raw): {compute_f1(label_pairs):.4f}")
    return label_pairs


if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repeat", type=int, default=1)
    parser.add_argument("--multiprocessing", action="store_true", default=True)
    parser.add_argument("--max_workers", type=int, default=48)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--expr_name", type=str, default="em_gpt5mini_results")
    parser.add_argument("--n_generation", type=int, default=15)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--eval_model", type=str, default="gpt-5-nano")
    parser.add_argument("--eval_reasoning_effort", type=str, default="minimal")
    parser.add_argument("--meta_reasoning_effort", type=str, default="none")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    # グローバル変数に反映（exec() 経由の生成エージェントコードから args にアクセスできないため）
    PRINT_LLM_DEBUG = args.debug
    EVAL_MODEL = args.eval_model
    EVAL_REASONING_EFFORT = args.eval_reasoning_effort
    META_REASONING_EFFORT = args.meta_reasoning_effort

    # search
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
