# app.py
import os
import json
import pathlib
import datetime as dt
import streamlit as st

from cli import GENRES, STYLES, TIMES, PLACES, CLUE_TYPES
from utils import AppState, Spec, render_judge_hud
from main import build_graph, judge_node  # ← 既存の main.py を利用

st.set_page_config(
    page_title="LLM推理オーケストレーション",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= サイドバー（実行条件） =================
st.sidebar.header("実行条件")
genre  = st.sidebar.selectbox("ジャンル", GENRES, index=GENRES.index("密室殺人"))
style  = st.sidebar.selectbox("作風", STYLES, index=STYLES.index("アガサクリスティ風"))
time_  = st.sidebar.selectbox("舞台の時間", TIMES, index=TIMES.index("冬の夜"))
place  = st.sidebar.selectbox("舞台の場所", PLACES, index=PLACES.index("音楽ホール"))

suspects_n = st.sidebar.number_input("容疑者数", min_value=2, max_value=6, value=3, step=1)
clues_n    = st.sidebar.number_input("証拠数",   min_value=3, max_value=10, value=3, step=1)
clue_types = st.sidebar.multiselect("証拠タイプ", CLUE_TYPES, default=["key","log","footstep"])

max_turns  = st.sidebar.number_input("最大ターン（3の倍数推奨）", min_value=3, max_value=24, value=9, step=3)
phase_init = st.sidebar.selectbox("開始フェーズ", ["整理","仮説","反論","結論"], index=0)

st.sidebar.divider()
show_judge = st.sidebar.checkbox("Judge HUD を表示", value=True)
log_path   = st.sidebar.text_input("Judge ログ保存（JSONL / 空で無効）", value="")

if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("⚠️ OPENAI_API_KEY が未設定です。環境変数に設定してください。")

# ================= セッション初期化 =================
def _init_session():
    for key, default in {
        "graph": None,
        "state": None,
        "spec": None,
        "run_log": [],
        "case_ready": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

_init_session()

# ================= 補助：JSONL追記 =================
def append_jsonl(path: str, obj: dict):
    if not path:
        return
    p = pathlib.Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **obj}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ================= 補助：事件生成をグラフで実行 =================
def run_until_case_generated():
    """
    エントリポイント（generate_case）→ facilitator までを実行し、
    state['case'] が埋まったら停止。run_log には何も積まない。
    """
    graph = st.session_state.graph
    state = st.session_state.state
    if not graph or state is None:
        return

    config = {"configurable": {"thread_id": "ui-run"}}
    for st_update in graph.stream(state, config=config, stream_mode="values"):
        st.session_state.state = st_update
        case = st_update.get("case") or {}
        if case:
            st.session_state.case_ready = True
            break

# ================= 補助：1ターンだけ進める =================
def step_once():
    """
    探偵の新規発話が1件出たら停止。
    その直後に judge_node を手動実行して数値ルーブリックを run_log に追加。
    """
    graph = st.session_state.graph
    state = st.session_state.state
    if not graph or state is None:
        return

    config = {"configurable": {"thread_id": "ui-run"}}
    prev_len = len(state.get("messages", []))
    got_new = False

    for st_update in graph.stream(state, config=config, stream_mode="values"):
        st.session_state.state = st_update

        # 探偵[A/B/C]の新規発言が出たら1ターン停止
        msgs = st_update.get("messages", [])
        if len(msgs) > prev_len:
            last = next((m for m in reversed(msgs) if m.get("role") in ["A","B","C"]), None)
            if last:
                st.session_state.run_log.append((last["role"], last["content"]))
                got_new = True
                break

        # 結論が生成されたら停止
        if st_update.get("final_verdict"):
            got_new = True
            break

    # Judge を回して HUD と ルーブリックを表示・保存
    if got_new and not st.session_state.state.get("final_verdict"):
        st.session_state.state = judge_node(st.session_state.state)
        j = st.session_state.state.get("judge", {}) or {}
        if show_judge:
            st.session_state.run_log.append(("JUDGE", render_judge_hud(j)))
            if "rubric" in j:
                st.session_state.run_log.append(("RUBRIC", j["rubric"]))
        if log_path:
            append_jsonl(log_path, {"judge": j})

    # 結論が出ていればログにも載せる
    if st.session_state.state.get("final_verdict"):
        fv = st.session_state.state["final_verdict"]
        st.session_state.run_log.append(("FINAL", json.dumps(fv, ensure_ascii=False)))

# ================= 上部ボタン =================
colA, colB, colC, colD = st.columns([1,1,1,2])
with colA:
    gen_btn = st.button("① 事件を生成", use_container_width=True)
with colB:
    step_btn = st.button("② 1ターン進める", use_container_width=True)
with colC:
    auto_btn = st.button("③ オート実行（最後まで）", use_container_width=True)
with colD:
    reset_btn = st.button("リセット", use_container_width=True, type="secondary")

# ================= リセット =================
if reset_btn:
    for k in list(st.session_state.keys()):
        if k in ("graph","state","spec","run_log","case_ready"):
            del st.session_state[k]
    _init_session()
    st.rerun()

# ================= ① 事件を生成 =================
if gen_btn:
    # Spec を構築
    spec: Spec = {
        "schema_version": "1.0",
        "outline": {"genre": genre, "style": style, "stage": {"time": time_, "place": place}},
        "suspects_spec": {"count": int(suspects_n)},
        "clues_spec": {"count": int(clues_n), "type_enum": clue_types},
        "constraints_spec": [],
    }
    st.session_state.spec = spec

    # グラフを用意（エントリは generate_case）
    st.session_state.graph = build_graph()

    # 初期状態（case は空で開始。グラフで生成させる）
    init: AppState = {
        "spec": spec,
        "case": {},                 # ← ここは空（generate_case ノードで生成）
        "messages": [],
        "turn": 0,
        "phase": phase_init,        # 自動切替は main.py の decide_phase に依存
        "speaker": "F",
        "queue": [],
        "final_verdict": None,
        "max_turns": int(max_turns),
        "round": 0,
        "judge": {},
    }
    st.session_state.state = init
    st.session_state.run_log = []
    st.session_state.case_ready = False

    # 事件が生成されるまで回して停止
    with st.spinner("事件を生成中…"):
        run_until_case_generated()

    # 生成直後に Judge 初期評価（任意）
    if st.session_state.case_ready:
        st.session_state.state = judge_node(st.session_state.state)
        j = st.session_state.state.get("judge", {}) or {}
        if show_judge:
            st.session_state.run_log.append(("JUDGE", render_judge_hud(j)))
            if "rubric" in j:
                st.session_state.run_log.append(("RUBRIC", j["rubric"]))
        if log_path:
            append_jsonl(log_path, {"judge": j})

# ================= ② 1ターン進める =================
if step_btn:
    if st.session_state.graph and st.session_state.state:
        if not st.session_state.case_ready:
            st.warning("先に『① 事件を生成』を押してください。")
        else:
            step_once()
    else:
        st.warning("先に『① 事件を生成』を押してください。")

# ================= ③ オート実行 =================
if auto_btn:
    if st.session_state.graph and st.session_state.state:
        if not st.session_state.case_ready:
            st.warning("先に『① 事件を生成』を押してください。")
        else:
            safety = 300
            while safety > 0:
                if st.session_state.state.get("final_verdict"):
                    break
                step_once()
                safety -= 1
    else:
        st.warning("先に『① 事件を生成』を押してください。")

# ================= 画面：表示 =================
st.header("事件仕様 / ケース")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Spec")
    st.json(st.session_state.spec or {})
with col2:
    st.subheader("Case（生成結果）")
    st.json((st.session_state.state or {}).get("case", {}) if st.session_state.state else {})

st.header("ディスカッション")
if st.session_state.run_log:
    for role, content in st.session_state.run_log:
        if role == "JUDGE":
            st.markdown(
                f"<pre style='background:#0b1020;color:#8fe;padding:10px;border-radius:8px'>{content}</pre>",
                unsafe_allow_html=True
            )
        elif role == "RUBRIC":
            # content は dict（rubric）
            order = [
                "coherence","evidence_alignment","novelty","counter_argument",
                "specificity","evidence_consistency","balance","global_score"
            ]
            display = {
                "coherence": "論旨の一貫性",
                "evidence_alignment": "証拠整合",
                "novelty": "新規性",
                "counter_argument": "反論の質",
                "specificity": "具体性",
                "evidence_consistency": "証拠の整合",
                "balance": "手口/動機/機会",
                "global_score": "総合",
            }
            data = [{"評価軸": display.get(k,k), "スコア(0-5)": float(content.get(k, 0.0))} for k in order]
            st.table(data)

        elif role == "FINAL":
            st.markdown("**[FINAL] 最終結論（JSON）**")
            try:
                st.json(json.loads(content))
            except Exception:
                st.write(content)
        else:
            st.markdown(f"**[{role}]**")
            st.write(content)
        st.divider()
else:
    st.caption("まだ発話はありません。『② 1ターン進める』または『③ オート実行』を押してください。")

debug = st.sidebar.checkbox("Judgeデバッグを表示", value=False)
# …画面の一番下あたりに…
if debug and st.session_state.state and st.session_state.state.get("__judge_debug"):
    with st.sidebar.expander("🔎 Judge Debug", expanded=True):
        dbg = st.session_state.state["__judge_debug"]
        st.caption("Prompt")
        st.code(dbg.get("prompt",""))
        st.caption("Raw (LLM生出力)")
        st.code(dbg.get("raw",""))
        st.caption("Parsed (LLMを素直にJSON化)")
        st.json(dbg.get("raw_parsed", {}))
        st.caption("Final rubric (正規化後)")
        st.json(dbg.get("final_rubric", {}))


# === Judge quick diagnostics ===
def _all_zero_rubric(r: dict) -> bool:
    keys = ["coherence","evidence_alignment","novelty","counter_argument",
            "specificity","evidence_consistency","balance","global_score"]
    try:
        return all(float(r.get(k, 0.0)) == 0.0 for k in keys)
    except Exception:
        return True

def _judge_diagnostics(state):
    dbg = state.get("__judge_debug", {}) or {}
    raw = dbg.get("raw", "") or ""
    prompt = dbg.get("prompt", "") or ""
    judge = state.get("judge", {}) or {}
    rubric = judge.get("rubric", {}) or {}

    det_logs = [m for m in state.get("messages", []) if m.get("role") in ["A","B","C"]]
    target_present = bool(det_logs and det_logs[-1].get("content"))

    has_json_in_raw = ("{" in raw and "}" in raw)
    has_rubric_in_raw = ("rubric" in raw)
    parsed_has_rubric = isinstance(rubric, dict) and len(rubric) > 0
    all_zero = _all_zero_rubric(rubric)

    # 文字数/トークン上限の目安チェック
    diag = {
        "det_logs_count": len(det_logs),
        "target_present": target_present,
        "prompt_chars": len(prompt),
        "raw_chars": len(raw),
        "raw_has_json_braces": has_json_in_raw,
        "raw_mentions_rubric": has_rubric_in_raw,
        "parsed_has_rubric": parsed_has_rubric,
        "rubric_all_zero": all_zero,
        "parsed_keys": sorted(list(judge.keys())),
        "rubric_keys": sorted(list(rubric.keys())) if parsed_has_rubric else [],
    }
    return diag

st.divider()
if st.session_state.state and st.session_state.state.get("__judge_debug"):
    st.subheader("Judge Quick Diagnostics")
    st.json(_judge_diagnostics(st.session_state.state))


