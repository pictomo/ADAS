# エンティティマッチング (AbtBuy) タスク用のプロンプト定義モジュール
# エージェント探索に使用するプロンプトテンプレート、初期アーカイブ、リフレクション用プロンプトを定義する

import json

# エージェント出力のフォーマット例（メタプロンプト内でLLMに提示するテンプレート）
# thought: エージェント設計の着想・全体構想・実装手順
# name: エージェントの名称
# code: forward()メソッドの実装コード
EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
""",
}

# ===== 初期アーカイブ: 探索の出発点となるベースラインエージェント群 =====

# Chain-of-Thought (CoT) エージェント
# LLMにフィールドごとの比較を段階的に行わせた後、マッチ判定を得る基本的な手法
COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables systematic field-by-field comparison of entity pairs. This practice improves the model's ability to handle tricky near-duplicate products and provides insight into its decision-making process.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step, comparing the two entities field by field (name, description, manufacturer, price), and then decide if they refer to the same real-world product. Output true for match, false for non-match."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], cot_instruction)
    return answer
""",
}

# Self-Consistency with CoT エージェント
# 複数のCoTエージェントを独立実行し、多数決で最終判定を決定する手法
COT_SC = {
    "thought": "While an LLM can arrive at the correct match decision, its reasoning may vary. By repeatedly asking the same question independently, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final decision through majority voting.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then decide if the two entities refer to the same real-world product. Output true for match, false for non-match."
    N = 5

    # 独立した複数エージェントで多様な推論経路を生成
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent') for _ in range(N)]

    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]

    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    # 複数エージェントの回答を多数決で集約
    answer = majority_voting(possible_answers)
    return answer
""",
}

# Self-Refine (Reflexion) エージェント
# 批評エージェントのフィードバックを基に、LLMが反復的にマッチ判定を改善していく手法
# Critic Agentが正確と判断したら早期終了する
Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its entity matching decision based on feedback. A critic agent reviews the match decision and highlights potentially overlooked field-level evidence. The model then refines its reasoning and provides a more accurate decision.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    cot_initial_instruction = "Please think step by step, compare the two entities field by field, and decide if they refer to the same real-world product. Output true for match, false for non-match."
    cot_reflect_instruction = "Given previous attempts and feedback, carefully reconsider your entity matching decision. Pay close attention to any field-level evidence (name similarity, description overlap, price, manufacturer) you may have overlooked. Try to make a better-reasoned decision."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    critic_instruction = "Please review the entity matching decision above and criticize where it might be wrong. Check if any important field signals were overlooked or misinterpreted. If you are absolutely sure the decision is correct, output 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')

    N_max = 5

    # 初期判定
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # 批評エージェントによるフィードバックと正誤確認
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break

        # フィードバックを入力に追加して次の反復へ
        cot_inputs.extend([thinking, answer, feedback])
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
""",
}

# LLM Debate エージェント
# 異なるドメインの専門家役（e-commerce専門家・データ品質アナリスト・家電スペシャリスト）が議論し、
# 多角的な視点から最良のマッチ判定を導き出す手法
LLM_debate = {
    "thought": "By letting different LLMs with domain-specific roles debate the entity matching decision, we can leverage diverse perspectives. An e-commerce expert focuses on product taxonomy, a data quality analyst on identifier consistency, and a consumer electronics specialist on technical specifications.",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    debate_initial_instruction = "Please think step by step and then decide if the two entities refer to the same real-world product. Output true for match, false for non-match."
    debate_instruction = "Given decisions and reasoning from other agents, consider their perspectives as additional evidence. Please think carefully and provide an updated match/non-match decision."

    # EM向けのドメイン専門家ロールを設定
    roles = ['E-commerce Product Expert', 'Data Quality Analyst', 'Consumer Electronics Specialist']
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', role=role) for role in roles]

    final_decision_instruction = "Given all the above reasoning and match/non-match decisions, weigh the evidence carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent')

    max_round = 2
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # 各ラウンドで全エージェントが順番に議論する
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                # 他エージェントの前ラウンドの推論を参照して更新
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)

    # 全議論を踏まえて最終判定エージェントが結論を出す
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
""",
}

# Quality-Diversity (QD) エージェント
# 品質多様性手法に着想を得て、異なるアプローチで複数のエンティティ分析を生成し、
# 最終的に集約してマッチ判定を導き出す手法
QD = {
    "thought": "Similar to Quality-Diversity methods, generating multiple diverse analyses of an entity pair can help catch different matching signals. Each iteration is prompted to approach the comparison differently, and a final decision agent aggregates the diverse analyses.",
    "name": "Quality-Diversity",
    "code": """def forward(self, taskInfo):
    cot_initial_instruction = "Please think step by step and then decide if the two entities refer to the same real-world product. Output true for match, false for non-match."
    qd_instruction = "Given previous analyses of this entity pair, try a completely different approach or focus on different fields to analyze whether the two entities match. Output true for match, false for non-match."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    final_decision_instruction = "Given all the above analyses and match/non-match decisions, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent')

    N_max = 3
    cot_inputs = [taskInfo]
    possible_answers = []

    # 初回分析
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # 前回の分析を参照しつつ、異なる観点からの分析を追加
        cot_inputs.extend([thinking, answer])
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # 多様な分析を集約して最終判定
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
""",
}

# Dynamic Role Assignment エージェント
# Auto-GPTやExpert Promptingに着想を得て、エンティティペアの内容をルーティングエージェントが判定し、
# 最適な専門家エージェントに動的に割り当てる手法
Role_Assignment = {
    "thought": "Similar to expert prompting, we use a routing agent to dynamically select the most appropriate specialist for the entity pair. Different product categories may require different expertise to identify subtle matches or differences.",
    "name": "Dynamic Assignment of Roles",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step, compare the two entities field by field, and decide if they refer to the same real-world product. Output true for match, false for non-match."

    roles = ['E-commerce Product Expert', 'Consumer Electronics Specialist', 'Data Quality Analyst', 'Helpful Assistant']
    expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in roles]

    # タスクに最適な専門家を選択するルーティングエージェント
    routing_instruction = "Given the two product entities, which specialist would be best suited to determine if they refer to the same product? Choose from: E-commerce Product Expert, Consumer Electronics Specialist, Data Quality Analyst."
    routing_agent = LLMAgentBase(['choice'], 'Routing Agent')

    choice = routing_agent([taskInfo], routing_instruction)[0]

    if 'e-commerce' in choice.content.lower() or 'product expert' in choice.content.lower():
        expert_id = 0
    elif 'electronics' in choice.content.lower():
        expert_id = 1
    elif 'data quality' in choice.content.lower() or 'analyst' in choice.content.lower():
        expert_id = 2
    else:
        expert_id = 3  # デフォルトはHelpful Assistant

    thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
    return answer
""",
}

# LLMへのシステムプロンプト（JSON形式での応答を強制する）
system_prompt = (
    """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""
)

# メタプロンプト本体: エージェント探索のためのメインプロンプトテンプレート
# [ARCHIVE] にはこれまでに発見されたエージェントアーキテクチャのアーカイブが挿入される
# [EXAMPLE] には出力フォーマットの例が挿入される
# 構成:
#   1. 概要説明（エンティティマッチングタスクの説明、ユーティリティコードの参照情報）
#   2. 発見済みアーキテクチャのアーカイブ
#   3. 出力フォーマットの指示と例
#   4. よくある実装ミスの例（注意喚起）
#   5. 新しいエージェント設計のタスク指示
base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the **Entity Matching** task: given a pair of product descriptions from two different e-commerce catalogs, determine whether they refer to the same real-world product.

## An example question from the Entity Matching task:

**Entity Pair**:
Entity A:
  Name: Sony Turntable - PSLX350H
  Description: Sony Turntable - PSLX350H/ Belt Drive System/ 33-1/3 and 45 RPM Speeds/ Servo Speed Control/ Supplied Moving Magnet Phono Cartridge/ Bonded Diamond Stylus/ Static Balance Tonearm/ Pitch Control
  Price: $179.00

Entity B:
  Name: Sony PS-LX350H Automatic Turntable
  Description: Belt drive automatic turntable, 33-1/3 and 45 RPM, built-in phono pre-amplifier
  Manufacturer: Sony
  Price: $149.99

**Answer (Not Given)**: true (same product, different catalog listings)

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\\n{str(request_keys)}\\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        reasoning_effort=EVAL_REASONING_EFFORT,
        max_completion_tokens=[MAX_COMPLETION_TOKENS],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.

    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used. (option. Keep it default.)
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model=None) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model if model is not None else '[EVAL_MODEL]'
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        Generates a prompt for the LLM.

        An example of a generated prompt:
        ""
        You are a helpful assistant.

        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        Entity A: ...
        Entity B: ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        # Instruction:
        Please think step by step and then decide if the two entities match.
        ""
        \"""
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY true (match) or false (non-match)." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\\n\\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\\n{content}\\n\\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\\n{content}\\n\\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\\n{content}\\n\\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.

        Args:
        - taskInfo (Info): Task information.

        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
          The answer should be true (match) or false (non-match).
        \"""
        pass
```
# Model Constraints
The LLMAgentBase calls use model `[EVAL_MODEL]` with `reasoning_effort="[EVAL_REASONING_EFFORT]"` and `max_completion_tokens=[MAX_COMPLETION_TOKENS]`. Keep these constraints in mind:
- **Calibrate complexity to `[EVAL_MODEL]` / `[EVAL_REASONING_EFFORT]`.** A small number of focused output fields (2-3) and shorter context chains are safer for weaker models or lower reasoning effort, which risk incomplete JSON responses or shallow reasoning on long, multi-step prompts. A more capable model / higher reasoning effort can handle more output fields and longer chains.
- **Do not hardcode a model name.** Leave `model=None` to use the default.

# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the **F1 score** on a validation set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
Finally, the last key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture.
Also, it could be helpful to set the LLM's role to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Infos.content.
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\\n'.join(all_thinking)
aggregated_answers = '\\n'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)

# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" (F1 score) by proposing interestingly new agents for the entity matching task.
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX.
"""

# リフレクションプロンプト1: 生成されたエージェント設計を批判的に振り返るためのプロンプト
# 1. 革新性の評価 2. 実装ミスの特定 3. 改善提案 を行わせる
Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings.
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

# リフレクションプロンプト2: 「よくある実装ミス」を参照して更にコードを修正させるプロンプト
Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""


BASELINES = {
    "cot": COT,
    "sc": COT_SC,
    "reflexion": Reflexion,
    "debate": LLM_debate,
    "qd": QD,
    "role": Role_Assignment,
}


def get_prompt(
    current_archive, eval_model: str, eval_reasoning_effort: str, max_completion_tokens: int
):
    """探索用のメタプロンプトを生成する。

    メタプロンプトテンプレート(base)にアーカイブ・出力例・評価モデル名・推論量・最大トークン数を埋め込み、
    LLMに新しいエージェントアーキテクチャを提案させるためのプロンプトを構築する。

    Args:
        current_archive (list[dict]): これまでに発見されたエージェントのリスト。
        eval_model (str): エージェント評価に使用するモデル名。[EVAL_MODEL] プレースホルダに埋め込む。
        eval_reasoning_effort (str): 評価モデルの推論量。[EVAL_REASONING_EFFORT] プレースホルダに埋め込む。
        max_completion_tokens (int): エージェント評価APIコールの max_completion_tokens。[MAX_COMPLETION_TOKENS] プレースホルダに埋め込む。

    Returns:
        tuple: (システムプロンプト, ユーザープロンプト)
    """
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    prompt = prompt.replace("[EVAL_MODEL]", eval_model)
    prompt = prompt.replace("[EVAL_REASONING_EFFORT]", eval_reasoning_effort)
    prompt = prompt.replace("[MAX_COMPLETION_TOKENS]", str(max_completion_tokens))
    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    """リフレクション用のプロンプトペアを生成する。

    前回試行したエージェントの情報をリフレクションプロンプトに埋め込み、
    LLMに設計の振り返りと改善を促すプロンプトを構築する。

    Args:
        prev_example (dict | None): 前回試行したエージェントの情報。
            Noneの場合は前回例なしでプロンプトを生成する。

    Returns:
        tuple: (リフレクションプロンプト1, リフレクションプロンプト2)
    """
    prev_example_str = (
        "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    )
    r1 = (
        Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str)
        if prev_example
        else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    )
    return r1, Reflexion_prompt_2
