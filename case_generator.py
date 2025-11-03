# case_generator.py
"""
事件JSONをLLMで生成する最小ユーティリティ。
- 依存: langchain_openai, utils (自作)
- 入口: generate_case(spec, ...)
"""
from typing import Dict, Optional
import json
from langchain_openai import ChatOpenAI

from utils import (
    Spec,
    AppState,
    json_sanitize,
    remap_ids_and_normalize,
    validate_case,
)

# ---- LLM ラッパ ----

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.4, max_tokens: int = 800) -> ChatOpenAI:
    """
    OpenAI系モデルをLangChain経由で呼ぶ最小ラッパ。
    OPENAI_API_KEY は環境変数から自動取得されます。
    """
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

# ---- プロンプト ----

def build_case_prompt(spec: Spec) -> str:
    """
    仕様（Spec）に厳密一致する事件JSONを出させるプロンプト。
    """
    return (
        "あなたは事件生成エージェントです。以下の仕様に厳密一致する事件データを生成してください。\n"
        "【出力規則（厳守）】\n"
        "・出力はJSONのみ。説明文やマークダウンは禁止。\n"
        "・最後に 'END_CASE' を単独行で付けること。\n"
        "・トップレベルキー: outline, victim, time_window, constraints, suspects, clues, truth\n"
        "・outline.genre / outline.style / outline.stage.time / outline.stage.place は仕様と一致。\n"
        "・suspects は spec.suspects_spec.count 人。id は 'S1','S2',...。\n"
        "・clues は spec.clues_spec.count 件。id は 'C1','C2',...。\n"
        "・clues[*].type は spec.clues_spec.type_enum から選ぶ。\n"
        "・truth.critical_clues は既存 clues[*].id のみ。\n"
        "・全て日本語で自然に記述。\n\n"
        f"【仕様】\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n"
    )

# ---- 生成本体 ----

def generate_case(
    spec: Spec,
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: int = 500,
    retries: int = 2,
) -> Dict:
    """
    事件JSONを生成して返します。
    - sanitize → json.loads → 正規化 → validate の順に処理。
    - 軽いリトライ付き（フォーマット崩れ対策）。
    失敗時は最後の例外を送出します。
    """
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    prompt = build_case_prompt(spec)

    last_err: Optional[Exception] = None
    for _ in range(max(retries, 1)):
        try:
            resp = llm.invoke(prompt)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            text = json_sanitize(raw)
            case = json.loads(text)

            # ID標準化・キー整備
            case = remap_ids_and_normalize(case)

            # 仕様に合致するか検証
            validate_case(case, spec)
            return case

        except Exception as e:
            last_err = e
            # フォーマット崩れが多い場合は、次ループでそのまま再試行（プロンプトは同じ）
            continue

    # ここまで来たら失敗
    assert last_err is not None
    raise last_err

# ---- LangGraph----事件JSONを 生成→抽出→正規化→検証

def generate_case_node(state: AppState) -> AppState:
    """
    LangGraph 用: state['spec'] を入力に事件JSONを生成し、state['case'] に格納して返す。
    例外は上位（グラフ側）で捕捉してください。
    """
    spec = state["spec"]
    case = generate_case(spec)
    state["case"] = case
    # フェーズ管理は外側の方針に従う（ここでは変更しない）
    return state
