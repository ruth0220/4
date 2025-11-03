# graph_main.py
"""
LangGraph の起動スクリプト（最小版）
- フェーズ自動切替なし（外部/手動で state["phase"] を設定する想定）
"""
import json
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from cli import parse_args, CLUE_TYPES
from utils import (
    Spec, AppState, Message,
    choose_next_speaker_with_judge, decide_phase,json_sanitize,extract_footer_json,)
from case_generator import get_llm, generate_case_node

#探偵
def detective_node(role: Literal["A", "B", "C"], style_hint: str):
    """
    探偵A/B/Cの発話を1ターン進める。
    - 直近の探偵ログ（最大8件）を与えて冗長化を抑制
    - 初手は所見から、以後は直前発言への自然な賛否＋自説
    """
    def _node(state: AppState) -> AppState:
        case = state["case"]
        messages = state.get("messages", [])
        det_logs = [m for m in messages if m.get("role") in ["A", "B", "C"]]
        conv = "\n".join(f"{m['role']}: {m['content']}" for m in det_logs[-8:])

        prev = det_logs[-1] if det_logs else None
        has_prev = prev is not None

        if not has_prev:
            prompt = (
                f"あなたは探偵{role}。{style_hint}\n"
                "本文では証拠の参照は内容で述べ、ID（C1等）は書かない。\n"
                "初手ルール：他者への“反応”で始めず、自分の所見から自然に始める。\n"
                "出力要件：箇条書きにせず『所見→根拠』の順に短く整然と書く。既出の内容を繰り返さない。\n"
                "【必須】: 直近の主張一覧を読み、そこに無い新規ポイントを最低1つ提示する。\n"
                "【必須】:直近の主張から1点を選び、具体的根拠を挙げて支持または反駁する（引用は短く）。\n"
                f"現在フェーズ: {state.get('phase','整理')}\n"
                f"事件データ（JSON）:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
                f"直近ログ（探偵のみ）:\n{conv}\n"
            )
        else:
            prompt = (
                f"あなたは探偵{role}。{style_hint}\n"
                "本文では証拠の参照は内容で述べ、ID（C1等）は書かない。\n"
                "出力要件：\n"
                "1) 最初の一文は直前発言への自然な賛否・反応。\n"
                "2) 『自分の意見→根拠』の順で、直前の主張に最低1点は具体的に言及（短く引用可）。\n"
                "3) 箇条書き禁止。冗長な繰り返し禁止。\n"
                f"前の発言者: {prev['role']}\n"
                f"その内容:\n<<<\n{prev['content']}\n>>>\n"
                f"現在フェーズ: {state.get('phase','整理')}\n"
                f"事件データ（JSON）:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
                f"直近ログ（探偵のみ）:\n{conv}\n"
            )

        text = get_llm().invoke(prompt).content  # langchain_openai.ChatOpenAI
        content = text if isinstance(text, str) else str(text)

        # 状態更新
        state.setdefault("messages", []).append(Message(role=role, content=content.strip()))
        state["speaker"] = "F"
        state["turn"] = state.get("turn", 0) + 1
        return state
    return _node

A_NODE = detective_node("A", "冷静で論理的に矛盾を突く。")
B_NODE = detective_node("B", "直感型で大胆な仮説を先に述べる。")
C_NODE = detective_node("C", "観察重視で証拠の具体描写から組み立てる。")

#ファシリテータ 
# 置き換え版: ファシリテータ（審判駆動）
def facilitator_node(state: AppState) -> AppState:
    """
    役割:
      1) 探偵の直近発話からフッタを回収（used_clues / vote / confidence）
      2) Judge（別ノード）が state['judge'] を更新した前提で、フェーズを decide_phase で確定
      3) 終了条件を判定（phase=='結論' or max_turns 到達 or 高コンセンサス）
      4) 最終JSONを厳密抽出（崩れ対策＆footer_bufferによる used_clues 補完）
      5) 次の話者を judge 駆動で選択
    """
    max_turns = state.get("max_turns", 8)
    now_turn  = state.get("turn", 0)
    phase     = state.get("phase", "整理")

    # --- (1) 直近探偵発話のフッタ回収（任意だが推奨） ---
    det_logs = [m for m in state.get("messages", []) if m.get("role") in ["A", "B", "C"]]
    if det_logs:
        last = det_logs[-1]
        try:
            footer, _ = extract_footer_json(last["content"])
        except Exception:
            footer = {}
        if footer:
            buf = state.setdefault("footer_buffer", [])
            buf.append({"role": last["role"], **footer})

    # --- (2) Judge 提案をもとにフェーズ確定（ヒステリシス内蔵） ---
    #   ※ judge_node はグラフ上で A/B/C の後に必ず呼ばれる想定。
    #     もしグラフに未導入でも、ここは安全に現在値を使って動く。
    state["phase"] = decide_phase(state)
    phase = state["phase"]

    # --- 終了判定（phase か ターン か 収束度） ---
    judge     = state.get("judge", {}) or {}
    consensus = float(judge.get("consensus", 0.0))
    early_stop = (consensus >= 0.75)  # 充分な合意なら早期終了

    if phase == "結論" or now_turn >= max_turns or early_stop:
        case = state["case"]
        conv = "\n".join(f"{m['role']}: {m['content']}" for m in det_logs[-12:])
        footer_buf = state.get("footer_buffer", [])

        # 票/証拠の事前集計（フォールバック用）
        votes = [b.get("vote_culprit") for b in footer_buf if isinstance(b, dict)]
        used_sets = [set(b.get("used_clues", [])) for b in footer_buf if isinstance(b, dict)]
        prior_used = sorted(set().union(*used_sets)) if used_sets else []

        # --- 最終結論プロンプト（used_clues をできるだけ ID で返させる） ---
        prompt = (
            "あなたはファシリテータ。以下の議論ログと事件データに基づき、"
            "最終結論のみを**厳密なJSON**で出力せよ。本文・説明文は禁止。\n"
            'フォーマット厳守：'
            '{"final_verdict":{"culprit":"","trick":"","confidence":0.0},'
            '"justification":[],"used_clues":[]}\n'
            "可能なら used_clues は実在する証拠ID（C1..）のみ列挙。\n"
            f"事件データ:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
            f"議論ログ（直近）:\n{conv}\n"
            f"参考（探偵の投票・信頼度・使用証拠の集計）:\n"
            f"{json.dumps(footer_buf, ensure_ascii=False)}\n"
        )
        raw  = get_llm(max_tokens=200).invoke(prompt).content
        text = raw if isinstance(raw, str) else str(raw)

        # --- 純JSON抽出 & サニタイズ ---
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s:e + 1]
        try:
            fv = json.loads(json_sanitize(text))
        except Exception:
            # フォールバック: 多数票と prior_used を採用
            maj = ""
            if votes:
                try:
                    from collections import Counter
                    maj = Counter([v for v in votes if v]).most_common(1)[0][0]
                except Exception:
                    maj = votes[0] if votes else ""
            fv = {
                "final_verdict": {"culprit": maj, "trick": "", "confidence": 0.5},
                "justification": [],
                "used_clues": prior_used
            }

        # --- used_clues が空なら footer から補完 ---
        if (not fv.get("used_clues")) and prior_used:
            fv["used_clues"] = prior_used

        state["final_verdict"] = fv
        state["phase"] = "done"
        state["speaker"] = "F"
        return state

    # --- (5) 次の話者を“審判駆動”で選ぶ（弱者救済で議論の偏りを抑制） ---
    state["speaker"] = choose_next_speaker_with_judge(state)
    return state

def build_judge_prompt(state: AppState) -> str:
    case = state["case"]
    det_logs = [m for m in state.get("messages", []) if m.get("role") in ["A","B","C"]]
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in det_logs[-12:])
    footer_buf = state.get("footer_buffer", [])

    guide = """あなたは審判（Judge）です。以下の議論ログ・事件データ・各探偵のフッタ要約に基づき、
現在の議論を数値で評価し、次フェーズ提案を行います。

【評価ルーブリック（0〜5点、0.5刻み可）】
1) coherence（論旨の一貫性）: 主張→根拠→結論の筋が通っているか。主語・対象・時刻が正しいか。減点: 話題飛び、論点先取、自己矛盾。
2) evidence_alignment（証拠整合）: 引用した事実（事件の記述）と主張が一致しているか。減点: 読み違い/存在しない事実。
3) novelty（新規性）: 直近発言にない情報・視点・推論ステップの追加。同じ証拠でも角度を変えたか。減点: 言い換えだけ/要点の反復。
4) counter_argument（反論の質）: 既存主張の弱点提示・代替説明・反証可能点の明示。減点: 抽象的反対/論点ずらし。
5) specificity（具体性）: 固有名・時刻・距離・動線・行為など具体情報の提示。減点: 抽象語の連打/「可能」「あり得る」だけ。
6) evidence_consistency（証拠の整合）: 複数証拠間の整合/矛盾を検査しているか。減点: 単証拠依存/矛盾放置。
7) balance（手口/動機/機会）: Means/Motive/Opportunity のカバー度とバランス。
8) global_score（総合）: 上記を総合した評価。

【出力は最後に厳密なJSONのみ（本文は任意）】
{
  "phase_suggestion":"整理|仮説|反論|結論",
  "agent_scores":{"A":0.0,"B":0.0,"C":0.0},
  "leader":"A|B|C|",
  "consensus":0.0,
  "reasoning_signals":{"novelty":0.0,"contradiction":0.0,"evidence_use":0.0},
  "rubric":{
    "coherence":0.0,
    "evidence_alignment":0.0,
    "novelty":0.0,
    "counter_argument":0.0,
    "specificity":0.0,
    "evidence_consistency":0.0,
    "balance":0.0,
    "global_score":0.0
  },
  "notes":[]
}
"""
    return (
        f"{guide}\n"
        f"事件データ:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
        f"議論ログ（直近12）:\n{conv}\n"
        f"フッタ集計（used_clues / vote / confidence）:\n{json.dumps(footer_buf, ensure_ascii=False)}\n"
    )


def judge_node(state: AppState) -> AppState:
    prompt = build_judge_prompt(state)
    # --- debug: 送信プロンプト保存
    state["__judge_debug"] = {"prompt": prompt}

    raw = get_llm(model="gpt-4o-mini", temperature=0.2, max_tokens=700).invoke(prompt).content
    text = raw if isinstance(raw, str) else str(raw)
    # --- debug: 生レスポンス保存
    state["__judge_debug"]["raw"] = text

    s, e = text.rfind("{"), text.rfind("}")
    judge = {}
    if s != -1 and e != -1 and e > s:
        try:
            judge = json.loads(text[s:e+1])
        except Exception:
            judge = {}

    # ★★★★★【差し込み①】パース直後（“LLMそのまま”の構造を保存）★★★★★
    state["__judge_debug"]["raw_parsed"] = judge


    judge.setdefault("phase_suggestion", state.get("phase","整理"))
    judge.setdefault("agent_scores", {"A":0.5, "B":0.5, "C":0.5})
    judge.setdefault("leader", max(judge["agent_scores"], key=judge["agent_scores"].get))
    judge.setdefault("consensus", 0.0)
    judge.setdefault("reasoning_signals", {"novelty":0.0, "contradiction":0.0, "evidence_use":0.0})
    judge.setdefault("notes", [])

    # ★ 8項目に統一
    default_rubric = {
        "coherence":0.0,
        "evidence_alignment":0.0,
        "novelty":0.0,
        "counter_argument":0.0,
        "specificity":0.0,
        "evidence_consistency":0.0,
        "balance":0.0,
        "global_score":0.0,
    }
    r = judge.get("rubric") or {}
    for k in list(default_rubric.keys()):
        try:
            default_rubric[k] = float(r.get(k, default_rubric[k]))
        except Exception:
            pass
    judge["rubric"] = default_rubric

    # ★★★★★【差し込み②】正規化“後”の最終rubricを保存★★★★★
    state["__judge_debug"]["final_rubric"] = judge.get("rubric", {})

    state["judge"] = judge
    hist = state.setdefault("judge_history", [])
    hist.append(judge)
    return state

#ルーター
def router(state: AppState) -> str:
    if state.get("phase") == "done":
        return END
    if state.get("turn", 0) > state.get("max_turns", 8):
        return END
    spk = state.get("speaker", "F")
    if spk == "F":
        return "facilitator"
    if spk in ("A", "B", "C"):
        return spk
    return "facilitator"

#グラフ構築 
def build_graph():
    g = StateGraph(AppState)

    # ノード登録
    g.add_node("generate_case", generate_case_node)  # import from case_generator
    g.add_node("facilitator", facilitator_node)
    g.add_node("A", A_NODE)
    g.add_node("B", B_NODE)
    g.add_node("C", C_NODE)
    g.add_node("judge", judge_node) 

    # エントリ → 事件生成 → ファシリテータ
    g.set_entry_point("generate_case")
    g.add_edge("generate_case", "facilitator")

    # ファシリテータから分岐（A/B/C/自身 or END）
    g.add_conditional_edges(
        "facilitator", router,
        {"A": "A", "B": "B", "C": "C", "facilitator": "facilitator", END: END}
    )
    # 探偵は話したら必ずファシリテータへ戻る
    for det in ["A", "B", "C"]:
        g.add_edge(det, "facilitator")

    return g.compile(checkpointer=MemorySaver())

#エントリポイント 
if __name__ == "__main__":
    # 1) CLI引数
    args = parse_args()
    clue_types = [t.strip() for t in args.clue_types.split(",") if t.strip()]
    invalid = [t for t in clue_types if t not in CLUE_TYPES]
    if invalid:
        raise ValueError(f"Unknown clue types: {invalid}. Candidates: {CLUE_TYPES}")

    # 2) Spec 構築
    spec: Spec = {
        "schema_version": "1.0",
        "outline": {
            "genre": args.genre,
            "style": args.style,
            "stage": {"time": args.time, "place": args.place},
        },
        "suspects_spec": {"count": int(args.suspects)},
        "clues_spec": {"count": int(args.clues), "type_enum": clue_types},
        "constraints_spec": [],
    }

    # 3) グラフ生成
    graph = build_graph()

    # 4) 初期状態（フェーズ自動管理はしない）
    init: AppState = {
        "spec": spec,
        "case": {},
        "messages": [],
        "turn": 0,
        "phase": "整理",      # 手動で変更したい場合はここを変える
        "speaker": "F",
        "queue": [],
        "final_verdict": None,
        "max_turns": 8,      # ターン上限（到達したら結論を出して終了）
        "round": 0,          # 使わないなら参照しなくてOK
        "judge": {},
    }

    # 5) 実行（逐次表示）
    print("\n[RUN GRAPH]\n")
    config = {"configurable": {"thread_id": "run-0001"}}

    last_printed = None
    for st in graph.stream(init, config=config, stream_mode="values"):
        # A/B/C の最後の発話だけ逐次表示
        last = next((m for m in reversed(st.get("messages", [])) if m.get("role") in ["A", "B", "C"]), None)
        if last and last is not last_printed:
            print(f"[{last['role']}]\n{last['content']}\n")
            last_printed = last

        if st.get("final_verdict"):
            print("\n[FINAL VERDICT]\n", json.dumps(st["final_verdict"], ensure_ascii=False, indent=2))
            break

    print("\n[DONE]\n")


