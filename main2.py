import os
import asyncio, json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient as ModelClient

import json

# 事件スキーマを Python の辞書で定義
SCHEMA_DICT = {
    "meta": {"genre": "密室殺人", "style": "アガサクリスティ風",
             "stage": {"time": "冬の夜", "place": "音楽ホール"}},
    "victim": {"name": "M氏", "role": "ピアニスト"},
    "time_window": {"start": "18:00", "end": "19:00", "estimated_death": "18:30"},
    "constraints": ["内側サムターン施錠", "二重ロック窓", "床は絨毯"],
    "suspects": [
        {"id": "A", "profile": "演奏助手,30代", "alibi": "17:50退出記録", "notes": "口論目撃あり"},
        {"id": "B", "profile": "マネージャー,40代", "alibi": "18:40入室し通報", "notes": "管理権限カード所持"},
        {"id": "C", "profile": "スタッフ,20代", "alibi": "18:10廊下掃除", "notes": "物音なし証言"}
    ],
    "clues": [
        {"id": "K1", "type": "key", "desc": "内側施錠サムターン"},
        {"id": "F1", "type": "footstep", "desc": "絨毯で足音残りにくい"},
        {"id": "L1", "type": "log", "desc": "カードキー出入記録"}
    ],
    "ground_truth": {
        "culprit": "B", "motive": "契約トラブル",
        "trick": "外からサムターン回し装置", "critical_clues": ["K1", "L1"]
    }
}

# ← ここが「json.dumpsで文字列化」
#    Python辞書(SCHEMA_DICT) → JSONテキスト(CASE_SCHEMA) に変換
CASE_SCHEMA = json.dumps(SCHEMA_DICT, ensure_ascii=False, indent=2)

# --- モデル設定（軽めでOK）
def model():
    return ModelClient(model="gpt-4o-mini", max_output_tokens=400,api_key=os.getenv("OPENAI_API_KEY"))

# --- 事件生成
def build_case_generator():
    return AssistantAgent(
        name="CaseGen",
        model_client=model(),
        system_message=(
            "あなたは事件生成エージェント。以下のJSONスキーマに**厳密一致**する事件を生成せよ。"
            "出力は**JSONのみ**。説明文やマークダウンは禁止。最後に 'END_CASE' を単独行で出力。\n\n"
            "【スキーマ】\n" + CASE_SCHEMA  # ← さっき作ったJSON文字列をそのまま連結
            
        ),
    )

async def generate_case(user_conditions: str) -> dict:
    gen = build_case_generator()
    task = f"条件: {user_conditions}\n上記条件に沿って事件JSONを1件生成せよ。"

    team = RoundRobinGroupChat(
        [gen],
        termination_condition=TextMentionTermination("END_CASE"),
        max_turns=2,
    )

    last_text = {"content": ""}  # 直近のテキストを保持する入れ物

    # streamをConsoleへ流しつつ、テキストだけ自分でも回収
    async def tee(stream):
        async for m in stream:
            txt = getattr(m, "content", None)
            if isinstance(txt, str):           # TextMessage だけ拾う
                last_text["content"] = txt
            yield m

    # Console は async イテレータを受け取る必要あり
    await Console(tee(team.run_stream(task=task)))

    # 直近のテキストから END_CASE までを抜き出してJSONパース
    content = last_text["content"].split("END_CASE")[0].strip()

    # 念のための保険：先頭と末尾の余計な文字を落とす
    # （普通は不要ですが、パース失敗時に役立ちます）
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 最後の { ... } を強引に抽出する簡易フォールバック
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise


# --- 推理チーム
def build_reasoning_team(case_json:dict):
    F = AssistantAgent(
        name="Facilitator",
        model_client=model(),
        system_message=(
            "あなたはファシリテーター。各探偵の議論を整理し、最後に最終結論JSONのみを出力せよ。"
            "最終結論JSONフォーマット："
            '{"final_verdict":{"culprit":"","trick":"","confidence":0.0},'
            '"justification":[],"used_clues":[],"LIGHTSIDE_WON":true}'
            "出力は**JSONのみ**、説明文禁止。"
            f"事件データ: ```json\n{json.dumps(case_json, ensure_ascii=False)}\n```"
        ),
    )
    A = AssistantAgent(
        name="DetectiveA",
        model_client=model(),
        system_message=("冷静な性格・論理担当。仮説→反証→絞り込み。"
                        "推論は 仮説→必要条件→矛盾点 の順で述べる。証拠は CLUE:ID で参照。")
    )
    B = AssistantAgent(
        name="DetectiveB",
        model_client=model(),
        system_message=("直感型の性格・異常検知担当。見落とし/奇策を提示。"
                        "推論は 仮説→必要条件→矛盾点 の順で述べる。証拠は CLUE:ID で参照")
    )
    C = AssistantAgent(
        name="DetectiveC",
        model_client=model(),
        system_message=("観察眼が鋭い。証拠と時間整合性を検証。"
                        "推論は 仮説→必要条件→矛盾点 の順で述べる。証拠は CLUE:ID で参照")
    )
    term = TextMentionTermination("LIGHTSIDE_WON|FINAL_VERDICT") | MaxMessageTermination(8)
    return RoundRobinGroupChat([F, A, B, C], termination_condition=term, max_turns=8)

async def main():
    # 1) 事件生成
    case = await generate_case("登場3人, ジャンル=密室殺人, 舞台=音楽ホール, 冬の夜, 作風=クリスティ風")
    print("\n[CASE LOADED]\n", json.dumps(case, ensure_ascii=False, indent=2))

    # 2) 推理対話（上限メッセージで強制終了も併用）
    team = build_reasoning_team(case)

    print("\n[REASONING]\n")
    await Console(team.run_stream(task="事件の真相を議論し、最後は最終結論JSONのみで締めよ。"))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n^C で中断しました。")
