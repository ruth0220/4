import os,json,asyncio #OpenAIのAPIキーを読む,リアルタイムで進む様子を見せる(エイシンキオ)
from autogen_agentchat.agents import AssistantAgent #エージェント達
from autogen_agentchat.teams import RoundRobinGroupChat #順番に回す
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination #議論終了woteigi
from autogen_agentchat.ui import Console #各エージェントの発話をターミナルに表示
from autogen_ext.models.openai import OpenAIChatCompletionClient as ModelClient #各エージェントにLLMを渡す
from cli import parse_args

SHOW_GEN_CONSOLE = False
SHOW_REASON_CONSOLE = True

# モデル設定
def model():
    return ModelClient(model="gpt-4o-mini", max_output_tokens=400,api_key=os.getenv("OPENAI_API_KEY"))

# 事件生成エージェント(cli.pyのspecから受け取る)
def build_case_generator(spec:dict):
    spec_json = json.dumps(spec, ensure_ascii=False, indent=2)
    return AssistantAgent(
        name="CaseGen",
        model_client=model(),
        system_message=(
            "あなたは事件生成エージェント。以下の仕様に一致する事件を生成せよ。"
            "【出力規則】\n"
            "出力はJSONのみで説明文やマークダウンは禁止。最後に 'END_CASE' を単独行で出力。\n"
            "- トップレベルのキー：outline, victim, time_window, constraints, suspects, clues, truth\n"
            "- outline.genre / outline.style / outline.stage.time / outline.stage.place は仕様と一致させる。\n"
            "- suspects は spec.suspects_spec.count 人。suspects[*].id は 'S1','S2',... の連番。\n"
            "- clues は spec.clues_spec.count 件。clues[*].id は 'C1','C2',... の連番。\n"
            "- clues[*].type は spec.clues_spec.type_enum のいずれかから選ぶ。\n"
            "- truth.critical_clues は existing clues[*].id のみ。\n"
            "- すべての値は日本語で自然に記述。\n\n"
            f"【仕様】\n{spec_json}"
        ),
    )

async def generate_case(spec:dict) -> dict:
    gen = build_case_generator(spec)
    task = "仕様に沿って事件JSONを1件生成せよ。"

    team = RoundRobinGroupChat(
        [gen],
        termination_condition=TextMentionTermination("END_CASE"),
        max_turns=2,
    )

    #streamをConsoleに流す
    last_text = {"content": ""}  

    async def tee(stream):
        async for m in stream:
            txt = getattr(m, "content", None)
            if isinstance(txt, str):           
                last_text["content"] = txt
            yield m

    stream = tee(team.run_stream(task=task))
    if SHOW_GEN_CONSOLE:
        await Console(stream)      
    else:
        async for _ in stream:      
            pass

    content = last_text["content"].split("END_CASE")[0].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 最後の { ... } を強引に抽出する簡易フォールバック
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise

def remap_ids_and_normalize(case: dict) -> dict:
    "IDゆれ・キーゆれを吸収し、S1.. / C1.. に揃える。critical_clues も追随。"
    c = json.loads(json.dumps(case))  # deep copy

    # suspects → S1..（保険）
    if "suspects" in c:
        for i, s in enumerate(c["suspects"], 1):
            s["id"] = f"S{i}"

    # clues → C1..（保険）＋ old→new マップ
    id_map = {}
    if "clues" in c:
        for i, clue in enumerate(c["clues"], 1):
            old = clue.get("id")
            new = f"C{i}"
            clue["id"] = new
            if old:
                id_map[old] = new

    # truth（旧ground_truthも吸収）＆ critical_clues 追随
    truth = c.get("truth") or c.get("ground_truth")
    if truth:
        if "critical_clues" in truth:
            truth["critical_clues"] = [id_map.get(x, x) for x in truth["critical_clues"]]
        c["truth"] = truth
        c.pop("ground_truth", None)

    # outline ← meta のゆれ吸収
    if "outline" not in c and "meta" in c:
        c["outline"] = c.pop("meta")

    return c

def validate_case(case: dict, spec: dict) -> None:
    "生成直後に壊れたレコードを弾く軽量バリデーション。失敗時は例外を投げる。"
    # 必須キー
    for k in ["outline", "victim", "time_window", "constraints", "suspects", "clues", "truth"]:
        if k not in case:
            raise ValueError(f"missing key: {k}")

    # outline の一致（値そのものをチェック）
    o, so = case["outline"], spec["outline"]
    if o.get("genre") != so.get("genre"):   raise ValueError("outline.genre mismatch")
    if o.get("style") != so.get("style"):   raise ValueError("outline.style mismatch")
    stg, sstg = o.get("stage", {}), so.get("stage", {})
    if stg.get("time")  != sstg.get("time"):  raise ValueError("outline.stage.time mismatch")
    if stg.get("place") != sstg.get("place"): raise ValueError("outline.stage.place mismatch")

    # suspects 数 & 連番
    n_sus = spec["suspects_spec"]["count"]
    if len(case["suspects"]) != n_sus:
        raise ValueError(f"suspects count must be {n_sus}")
    if [s.get("id") for s in case["suspects"]] != [f"S{i}" for i in range(1, n_sus+1)]:
        raise ValueError("suspects ids must be S1..S{n}")

    # clues 数 & 連番 & type の制約
    n_clu = spec["clues_spec"]["count"]
    allowed_types = set(spec["clues_spec"]["type_enum"])
    if len(case["clues"]) != n_clu:
        raise ValueError(f"clues count must be {n_clu}")
    if [c.get("id") for c in case["clues"]] != [f"C{i}" for i in range(1, n_clu+1)]:
        raise ValueError("clue ids must be C1..C{m}")
    for c in case["clues"]:
        if c.get("type") not in allowed_types:
            raise ValueError(f"unsupported clue type: {c.get('type')}")

    # critical_clues が実在IDのみ
    cids = {c["id"] for c in case["clues"]}
    for cid in case["truth"].get("critical_clues", []):
        if cid not in cids:
            raise ValueError(f"critical clue '{cid}' not in clues ids")

# 推理エージェント
def build_reasoning_team(case_json:dict):
    F = AssistantAgent(
        name="Facilitator",
        model_client=model(),
        system_message=(
            "あなたはファシリテーター。各探偵の議論を整理し、最後に最終結論を出力せよ。議論フェーズの登場役は DetectiveA / DetectiveB / DetectiveC / Facilitator のみ。"
            "最終結論フォーマット："
            '{"final_verdict":{"culprit":"","trick":"","confidence":0.0},'
            '"justification":[],"used_clues":[],"LIGHTSIDE_WON":true}'
            f"事件データ: ```json\n{json.dumps(case_json, ensure_ascii=False)}\n```"
        ),
    )
    A = AssistantAgent(
        name="DetectiveA",
        model_client=model(),
        system_message=("冷静な性格・論理担当。"
                        "推論は 仮説→根拠→矛盾点 の順で述べる。証拠は CLUE:ID で参照。")
    )
    B = AssistantAgent(
        name="DetectiveB",
        model_client=model(),
        system_message=("直感型の性格。"
                        "推論は 仮説→根拠→矛盾点 の順で述べる。証拠は CLUE:ID で参照。")
    )
    C = AssistantAgent(
        name="DetectiveC",
        model_client=model(),
        system_message=("観察眼が鋭い。"
                        "推論は 仮説→根拠→矛盾点 の順で述べる。証拠は CLUE:ID で参照。")
    )
    term = TextMentionTermination("LIGHTSIDE_WON|FINAL_VERDICT") | MaxMessageTermination(8)
    return RoundRobinGroupChat([F, A, B, C], termination_condition=term, max_turns=8)

async def main():
    # 1) cli.pyから仕様を受け取る
    args = parse_args() 
    clue_types = [t.strip() for t in args.clue_types.split(",") if t.strip()]    
    
    spec = {
        "schema_version": "1.0",
        "outline": {
            "genre": args.genre,
            "style": args.style,
            "stage": {"time": args.time, "place": args.place}
        },
        "suspects_spec": {"count": args.suspects},
        "clues_spec": {"count": args.clues, "type_enum": clue_types},
        "constraints_spec": []  # 必要になったらCLI側にフラグ追加で
    }

    # 2)事件生成
    case = await generate_case(spec)
    case = remap_ids_and_normalize(case)  # 保険でID揃え＆キーゆれ吸収
    validate_case(case, spec)             # 仕様違反はここで例外に

    print("\n[CASE LOADED]\n", json.dumps(case, ensure_ascii=False, indent=2))

    # 3) 推理
    team = build_reasoning_team(case)
    print("\n[REASONING]\n")

    stream = team.run_stream(task="事件の真相を議論し、最後は最終結論で締めよ。")

    if SHOW_REASON_CONSOLE:
        await Console(stream)       
    else:
        async for _ in stream:      
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n^C で中断しました。")
