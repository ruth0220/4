from uuid import uuid4
import os, json, re
from dataclasses import dataclass, field
from typing import Literal, TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from cli import parse_args, CLUE_TYPES

PHASE_SCHEDULE = [
    ("整理", 1),   # Round 1 まで
    ("仮説", 3),   # Round 2〜3
    ("反論", 5),   # Round 4〜5
    ("結論", 10**9),
]

#ラウンドを計算する、3ごとに次のフェースにする
def set_phase_by_round(state):
    det_msgs = sum(1 for m in state["messages"] if m["role"] in ["A","B","C"])
    round_now = det_msgs // 3 if det_msgs % 3 == 0 and det_msgs > 0 else (det_msgs // 3 + (1 if det_msgs else 0))
    state["round"] = round_now or 0

    phase = "整理"
    for name, until in PHASE_SCHEDULE:
        if state["round"] <= until:
            phase = name
            break
    if state["turn"] >= state.get("max_turns", 8) - 1:
        phase = "結論"
    state["phase"] = phase

def llm(model: str = "gpt-4o-mini", max_tokens: int = 400):
    return ChatOpenAI(model=model, temperature=0.4, max_tokens=max_tokens)

# 仕様
class Spec(TypedDict):
    schema_version: str
    outline: Dict
    suspects_spec: Dict
    clues_spec: Dict
    constraints_spec: List

class Message(TypedDict):
    role: Literal["facilitator","A","B","C"]
    content: str

class Verdict(TypedDict):
    culprit: str
    trick: str
    confidence: float

class AppState(TypedDict):
    spec: Spec
    case: Dict
    messages: List[Message]
    turn: int
    phase: Literal["setup","整理","仮説","反論","結論","done"]
    speaker: Literal["F","A","B","C"]
    queue: List[str]  
    final_verdict: Optional[Dict]
    max_turns: int

def json_sanitize(text: str) -> str:
    # コードブロック除去
    text = re.sub(r"```(json)?", "", text)
    # END_CASE より前だけ
    text = text.split("END_CASE")[0]
    # 先頭/末尾の余分を削る
    text = text.strip()
    # 大括弧位置を救済
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

#ID正規化 
def remap_ids_and_normalize(case: dict) -> dict:
    c = json.loads(json.dumps(case, ensure_ascii=False))
    if "suspects" in c:
        for i, s in enumerate(c["suspects"], 1):
            s["id"] = f"S{i}"
    id_map = {}
    if "clues" in c:
        for i, clue in enumerate(c["clues"], 1):
            old = clue.get("id")
            new = f"C{i}"
            clue["id"] = new
            if old:
                id_map[old] = new
    truth = c.get("truth") or c.get("ground_truth")
    if truth:
        if "critical_clues" in truth:
            truth["critical_clues"] = [id_map.get(x, x) for x in truth["critical_clues"]]
        c["truth"] = truth
        c.pop("ground_truth", None)
    if "outline" not in c and "meta" in c:
        c["outline"] = c.pop("meta")
    return c

#バリデーション
def validate_case(case: dict, spec: Spec) -> None:
    for k in ["outline","victim","time_window","constraints","suspects","clues","truth"]:
        if k not in case:
            raise ValueError(f"missing key: {k}")
    o, so = case["outline"], spec["outline"]
    if o.get("genre") != so.get("genre"):   raise ValueError("outline.genre mismatch")
    if o.get("style") != so.get("style"):   raise ValueError("outline.style mismatch")
    stg, sstg = o.get("stage", {}), so.get("stage", {})
    if stg.get("time") != sstg.get("time"):   raise ValueError("outline.stage.time mismatch")
    if stg.get("place") != sstg.get("place"): raise ValueError("outline.stage.place mismatch")
    n_sus = spec["suspects_spec"]["count"]
    if len(case["suspects"]) != n_sus:
        raise ValueError(f"suspects count must be {n_sus}")
    if [s.get("id") for s in case["suspects"]] != [f"S{i}" for i in range(1, n_sus+1)]:
        raise ValueError("suspects ids must be S1..S{n}")
    n_clu = spec["clues_spec"]["count"]
    allowed_types = set(spec["clues_spec"]["type_enum"])
    if len(case["clues"]) != n_clu:
        raise ValueError(f"clues count must be {n_clu}")
    if [c.get("id") for c in case["clues"]] != [f"C{i}" for i in range(1, n_clu+1)]:
        raise ValueError("clue ids must be C1..C{m}")
    for c in case["clues"]:
        if c.get("type") not in allowed_types:
            raise ValueError(f"unsupported clue type: {c.get('type')}")
    cids = {c["id"] for c in case["clues"]}
    for cid in case["truth"].get("critical_clues", []):
        if cid not in cids:
            raise ValueError(f"critical clue '{cid}' not in clues ids")

#LangGraphノード

# 1) 事件生成
def generate_case_node(state: AppState) -> AppState:
    spec = state["spec"]
    prompt = (
        "あなたは事件生成エージェント。以下の仕様に一致する事件を生成せよ。\n"
        "【出力規則】\n"
        "出力はJSONのみで説明文やマークダウンは禁止。最後に 'END_CASE' を単独行で出力。\n"
        "- トップレベルのキー：outline, victim, time_window, constraints, suspects, clues, truth\n"
        "- outline.genre / outline.style / outline.stage.time / outline.stage.place は仕様と一致。\n"
        "- suspects は spec.suspects_spec.count 人。suspects[*].id は 'S1','S2',...。\n"
        "- clues は spec.clues_spec.count 件。clues[*].id は 'C1','C2',...。\n"
        "- clues[*].type は spec.clues_spec.type_enum から選ぶ。\n"
        "- truth.critical_clues は既存 clues[*].id のみ。\n"
        "- すべて日本語で自然に記述。\n\n"
        f"【仕様】\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n"
    )
    resp = llm().invoke(prompt)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = json_sanitize(raw)
    case = json.loads(text)
    case = remap_ids_and_normalize(case)
    validate_case(case, spec)
    state["case"] = case#事件を保存
    state["phase"] = "整理"#最初はここから
    state["speaker"] = "F"
    return state

#AならB、BならC、CならAで回す
def choose_next_speaker(state):
    order = ["A","B","C"]
    last = next((m for m in reversed(state["messages"]) if m["role"] in order), None)
    if not last:
        return "A"
    i = order.index(last["role"])
    return order[(i + 1) % len(order)] #探偵のラベルを返す

# 2) ファシリテータ
def facilitator_node(state: AppState) -> AppState:
    # 1) まずラウンド数からフェーズを確定
    set_phase_by_round(state)

    # 2) 結論フェーズなら最終JSONだけ生成
    if state["phase"] == "結論":
        case = state["case"]
        conv = "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["messages"] if m["role"] in ["A","B","C"]][-12:]
        )
        prompt = (
            "あなたはファシリテータ。以下の議論ログと事件データに基づき、"
            "最終結論のみを厳密なJSONで出力せよ。本文や説明文は一切禁止。\n"
            'フォーマット: {"final_verdict":{"culprit":"","trick":"","confidence":0.0},"justification":[],"used_clues":[],"LIGHTSIDE_WON":true}\n'
            f"事件データ:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
            f"議論ログ:\n{conv}\n"
        )
        resp = llm().invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s:e+1]
        fv = json.loads(text)

        state["final_verdict"] = fv
        state["phase"] = "done"
        state["speaker"] = "F"
        return state

    # 3) 結論以外はLLMを使わずに次話者を決定
    state["speaker"] = choose_next_speaker(state)
    return state

# 3) 探偵 
def detective_node(role: Literal["A","B","C"], style_hint: str):
    def _node(state: AppState) -> AppState:
        case = state["case"]
        conv = "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["messages"] if m["role"] in ["A","B","C"]][-8:]
        )
        # 直前の探偵発言
        prev = next((m for m in reversed(state["messages"]) if m["role"] in ["A","B","C"]), None)
        has_prev_det = prev is not None
        prev_role = prev["role"] if prev else "(none)"
        prev_text = prev["content"] if prev else "(none)"

        if not has_prev_det:
            # 初手用プロンプト：反応で始めない、自分の所見から言う
            prompt = (
                f"あなたは探偵{role}。{style_hint}\n"
                "本文では clue.label のみを使い、ID(C1等)は書かない。\n"
                "初手ルール：相手への賛否や『私もそう思う』『それは違う』等の“反応”で始めず、"
                "自分の所見から自然に書き始める（例：『第一印象では…』『私はこう見る。』）。\n"
                "出力要件：文章は箇条書きにせず、『所見→根拠』の順で。\n"
                "既出の発言を繰り返さない。\n"
                f"事件データ:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
                f"現在フェーズ:{state['phase']}\n"
                f"直近ログ（探偵のみ）:\n{conv}\n"
            )
        else:
            # 通常ターン：直前発言に賛否＋反論で応答
            prompt = (
                f"あなたは探偵{role}。{style_hint}\n"
                "本文では clue.label のみを使い、ID(C1等)は書かない。\n"
                "出力要件：\n"
                "1) 最初の一文は直前発言への自然な反応や賛否、同意や反論などを様々なバリエーションで。\n"
                "2) 続けて『自分の意見→根拠』の順で、直前の主張に最低1点は具体的に言及（短く引用可）。\n"
                "3) 箇条書きにしない。\n"
                f"前の発言者: {prev_role}\n"
                f"その内容:\n<<<\n{prev_text}\n>>>\n"
                f"事件データ:\n```json\n{json.dumps(case, ensure_ascii=False)}\n```\n"
                f"現在フェーズ:{state['phase']}\n"
                f"直近ログ（探偵のみ）:\n{conv}\n"
            )
        resp = llm().invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        state["messages"].append({"role": role, "content": text.strip()})
        state["speaker"] = "F"
        state["turn"] += 1
        return state
    return _node

A_NODE = detective_node("A", "冷静・論理担当。")
B_NODE = detective_node("B", "直感型。口調がとても明るい。")
C_NODE = detective_node("C", "観察眼が鋭い。")

def router(state: AppState) -> str:
    if state["phase"] == "done":
        return END
    if state["turn"] >= state["max_turns"]:
        return END
    spk = state["speaker"]
    if spk == "F":
        return "facilitator"
    elif spk == "A":
        return "A"
    elif spk == "B":
        return "B"
    elif spk == "C":
        return "C"
    return "facilitator"

#グラフ
def build_graph():
    g = StateGraph(AppState)
    g.add_node("generate_case", generate_case_node)
    g.add_node("facilitator", facilitator_node)
    g.add_node("A", A_NODE)
    g.add_node("B", B_NODE)
    g.add_node("C", C_NODE)

    g.set_entry_point("generate_case")

    # ルーティング: generate_case -> facilitator -> (ABC) -> facilitator ...
    g.add_conditional_edges("facilitator", router, {"A":"A","B":"B","C":"C","facilitator":"facilitator",END:END})
    for det in ["A","B","C"]:
        g.add_edge(det, "facilitator")

    # generate後は facilitator へ
    g.add_edge("generate_case", "facilitator")

    return g.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    # 1) CLI引数を取得
    args = parse_args()
    clue_types = [t.strip() for t in args.clue_types.split(",") if t.strip()]

    # 2) 入力検証
    invalid = [t for t in clue_types if t not in CLUE_TYPES]
    if invalid:
        raise ValueError(f"Unknown clue types: {invalid}. Candidates: {CLUE_TYPES}")

    # 3) Spec構築
    spec: Spec = {
        "schema_version": "1.0",
        "outline": {
            "genre": args.genre,
            "style": args.style,
            "stage": {"time": args.time, "place": args.place},
        },
        "suspects_spec": {"count": args.suspects},
        "clues_spec": {"count": args.clues, "type_enum": clue_types},
        "constraints_spec": [],
    }

    # 4) グラフを用意
    graph = build_graph()

    # 5) 初期状態
    init: AppState = {
        "spec": spec,
        "case": {},
        "messages": [],
        "turn": 0,
        "phase": "setup",
        "speaker": "F",
        "queue": [],  
        "final_verdict": None,
        "max_turns": 8,
        "round":0,
    }

    # 6) 実行（逐次表示）
    print("\n[RUN GRAPH]\n")
    config = {"configurable": {"thread_id": "run-xxxx"}} 

    last_printed = None
    for st in graph.stream(init, config=config, stream_mode="values"):
        # A/B/C の最後の発話だけ拾う
        last = next((m for m in reversed(st["messages"]) if m["role"] in ["A","B","C"]), None)
        if last and last is not last_printed:
            print(f"[{last['role']}]\n{last['content']}\n")
            last_printed = last

        if st.get("final_verdict"):
            print("\n[FINAL VERDICT]\n", json.dumps(st["final_verdict"], ensure_ascii=False, indent=2))
            break
print("\n[DONE]\n")

