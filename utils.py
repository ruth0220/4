# utils.py (minimal, no phase-schedule)
from typing import TypedDict, Literal, List, Dict, Optional,Tuple
import json
import re,os

# ===== 型定義 =====

class Spec(TypedDict):
    schema_version: str
    outline: Dict                  # {"genre":..., "style":..., "stage":{"time":..., "place":...}}
    suspects_spec: Dict            # {"count": int}
    clues_spec: Dict               # {"count": int, "type_enum": List[str]}
    constraints_spec: List

class Message(TypedDict):
    role: Literal["facilitator", "A", "B", "C"]
    content: str

class Verdict(TypedDict):
    culprit: str
    trick: str
    confidence: float

class AppState(TypedDict, total=False):
    spec: Spec
    case: Dict
    messages: List[Message]
    turn: int
    phase: Literal["setup", "整理", "仮説", "反論", "結論", "done"]  # ← 手動/任意更新でOK
    speaker: Literal["F", "A", "B", "C"]
    queue: List[str]
    final_verdict: Optional[Dict]
    max_turns: int
    round: int  # 使わなければ参照しなくてOK
    judge: Dict
    
# ==== Judge 出力スキーマ ====
class JudgeState(TypedDict, total=False):
    phase_suggestion: Literal["整理", "仮説", "反論", "結論"]
    agent_scores: Dict[str, float]   # 例 {"A":0.62,"B":0.48,"C":0.57}
    leader: str                      # "A" | "B" | "C" | ""
    consensus: float                 # 0..1
    reasoning_signals: Dict[str, float]  # {"novelty":..., "contradiction":..., "evidence_use":...}
    notes: List[str]                 # 任意の短文メモ


# ===== 話者管理 =====

def choose_next_speaker(state: AppState) -> str:
    """直近の探偵発話から A→B→C の順で次話者を返す"""
    order = ["A", "B", "C"]
    for m in reversed(state.get("messages", [])):
        if m.get("role") in order:
            i = order.index(m["role"])  # type: ignore[index]
            return order[(i + 1) % 3]
    return "A"

# ===== 文字列整形 =====

def json_sanitize(text: str) -> str:
    """モデル出力からJSONだけを取り出す簡易サニタイザ。"""
    text = re.sub(r"```(?:json)?", "", text)
    text = text.split("END_CASE")[0]
    text = text.strip()
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e > s else text

# ===== 事件JSONの正規化/検証 =====

def remap_ids_and_normalize(case: Dict) -> Dict:
    """suspectsをS1..、cluesをC1..に振り直し。truth/ground_truthやmetaも統一。"""
    c = json.loads(json.dumps(case, ensure_ascii=False))
    if "suspects" in c and isinstance(c["suspects"], list):
        for i, s in enumerate(c["suspects"], 1):
            s["id"] = f"S{i}"
    id_map: Dict[str, str] = {}
    if "clues" in c and isinstance(c["clues"], list):
        for i, clue in enumerate(c["clues"], 1):
            old = clue.get("id")
            new = f"C{i}"
            clue["id"] = new
            if old:
                id_map[old] = new
    truth = c.get("truth") or c.get("ground_truth")
    if truth:
        if isinstance(truth.get("critical_clues"), list):
            truth["critical_clues"] = [id_map.get(x, x) for x in truth["critical_clues"]]
        c["truth"] = truth
        c.pop("ground_truth", None)
    if "outline" not in c and "meta" in c:
        c["outline"] = c.pop("meta")
    return c

def validate_case(case: Dict, spec: Spec) -> None:
    """仕様とケースの整合チェック。合わなければ ValueError。"""
    required = ["outline", "victim", "time_window", "constraints", "suspects", "clues", "truth"]
    for k in required:
        if k not in case:
            raise ValueError(f"missing key: {k}")
    o, so = case["outline"], spec["outline"]
    if o.get("genre") != so.get("genre"):
        raise ValueError("outline.genre mismatch")
    if o.get("style") != so.get("style"):
        raise ValueError("outline.style mismatch")
    stg, sstg = o.get("stage", {}), so.get("stage", {})
    if stg.get("time") != sstg.get("time"):
        raise ValueError("outline.stage.time mismatch")
    if stg.get("place") != sstg.get("place"):
        raise ValueError("outline.stage.place mismatch")
    n_sus = int(spec["suspects_spec"]["count"])
    sus = case.get("suspects", [])
    if not isinstance(sus, list) or len(sus) != n_sus:
        raise ValueError(f"suspects count must be {n_sus}")
    if [s.get("id") for s in sus] != [f"S{i}" for i in range(1, n_sus + 1)]:
        raise ValueError("suspects ids must be S1..S{n}")
    n_clu = int(spec["clues_spec"]["count"])
    clu = case.get("clues", [])
    if not isinstance(clu, list) or len(clu) != n_clu:
        raise ValueError(f"clues count must be {n_clu}")
    if [c.get("id") for c in clu] != [f"C{i}" for i in range(1, n_clu + 1)]:
        raise ValueError("clue ids must be C1..C{m}")
    allowed_types = set(spec["clues_spec"].get("type_enum", []))
    for c in clu:
        if c.get("type") not in allowed_types:
            raise ValueError(f"unsupported clue type: {c.get('type')}")
    truth = case.get("truth", {})
    cids = {c["id"] for c in clu}
    for cid in truth.get("critical_clues", []):
        if cid not in cids:
            raise ValueError(f"critical clue '{cid}' not in clues ids")

def decide_phase(state: AppState) -> str:
    cur = state.get("phase","整理")
    j: Dict = state.get("judge", {}) or {}
    sug = j.get("phase_suggestion", cur)
    consensus = float(j.get("consensus", 0.0))
    signals = j.get("reasoning_signals", {}) or {}
    novelty = float(signals.get("novelty", 0.0))
    contradiction = float(signals.get("contradiction", 0.0))
    evidence_use = float(signals.get("evidence_use", 0.0))
    t = state.get("turn", 0)
    max_turns = state.get("max_turns", 8)

    if t >= max_turns - 1:
        return "結論"

    if cur == "整理":
        if novelty >= 0.35 or sug in ("仮説","反論","結論"):
            return "仮説"
        return "整理"

    if cur == "仮説":
        if contradiction >= 0.25 or sug == "反論":
            return "反論"
        if consensus >= 0.6 and evidence_use >= 0.5:
            return "結論"
        return "仮説"

    if cur == "反論":
        if consensus >= 0.6 and contradiction <= 0.15:
            return "結論"
        if novelty >= 0.25 and contradiction >= 0.2:
            return "反論"
        if novelty < 0.15 and contradiction < 0.15:
            return "仮説"
        return "反論"

    if cur == "結論":
        return "結論"

    return str(sug)

def choose_next_speaker_with_judge(state: AppState) -> str:
    j = state.get("judge", {}) or {}
    scores: Dict[str, float] = j.get("agent_scores", {}) or {}
    # スコアが低い順（弱者救済で議論の偏りを抑える）
    order = sorted(["A","B","C"], key=lambda x: scores.get(x, 0.5))
    # 直近話者を避ける
    last_role = None
    for m in reversed(state.get("messages", [])):
        if m.get("role") in ("A","B","C"):
            last_role = m["role"]; break
    for r in order:
        if r != last_role:
            return r
    return order[0]

def extract_footer_json(text: str) -> Tuple[dict, str]:
    """
    探偵の自然文本文から、末尾のフッタJSONを抽出する。
    返り値: (footer_dict, body_text)
      - footer_dict: {"used_clues": [...], "vote_culprit": "S?", "confidence": 0.x} など
      - body_text:   フッタを除いた本文（自然文）
    """
    s, e = text.rfind("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            footer = json.loads(text[s:e+1])
            body = text[:s].rstrip()
            # 期待キーが無ければ無効扱い
            required = {"used_clues", "vote_culprit", "confidence"}
            if not required.issubset(set(footer.keys())):
                return {}, text
            # 型のゆらぎを軽く吸収
            if not isinstance(footer.get("used_clues"), list):
                footer["used_clues"] = []
            if not isinstance(footer.get("vote_culprit"), str):
                footer["vote_culprit"] = ""
            try:
                footer["confidence"] = float(footer.get("confidence", 0.0))
            except Exception:
                footer["confidence"] = 0.0
            return footer, body
        except Exception:
            pass
    return {}, text

# 既存 render_judge_hud に総合点を追記（末尾に global_score）
def render_judge_hud(j: Dict) -> str:
    if not j:
        return "[JUDGE] (no data)"
    ps   = j.get("phase_suggestion","?")
    cons = f"{float(j.get('consensus',0.0)):.2f}"
    sc   = j.get("agent_scores",{}) or {}
    a = f"{float(sc.get('A',0.0)):.2f}"
    b = f"{float(sc.get('B',0.0)):.2f}"
    c = f"{float(sc.get('C',0.0)):.2f}"
    sig = j.get("reasoning_signals",{}) or {}
    nov = f"{float(sig.get('novelty',0.0)):.2f}"
    ctr = f"{float(sig.get('contradiction',0.0)):.2f}"
    evi = f"{float(sig.get('evidence_use',0.0)):.2f}"
    lead = j.get("leader","")
    rb = j.get("rubric",{}) or {}
    g = f"{float(rb.get('global_score',0.0)):.2f}"
    notes = j.get("notes",[])
    line1 = f"[JUDGE] phase={ps} | consensus={cons} | leader={lead} | scores A:{a} B:{b} C:{c} | sig(nov/ctr/evi)={nov}/{ctr}/{evi} | global={g}"
    line2 = (f"        notes: { '; '.join(notes[:2]) }" if notes else "")
    return line1 + ("\n" + line2 if line2 else "")

# ルーブリックを UI 表示用の行リストに整形
def rubric_rows(j: Dict) -> List[Dict[str, float]]:
    r = j.get("rubric",{}) or {}
    keys = [
        "coherence","evidence_alignment","novelty","counter_argument",
        "specificity","evidence_consistency","balance","global_score"
    ]
    return [{"metric": k, "score": float(r.get(k, 0.0))} for k in keys]


def append_jsonl(path: str, obj: Dict) -> None:
    """指定パスにJSON Linesで1件追記。ディレクトリは自動作成。"""
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **obj}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

