import argparse

GENRES   = ["密室殺人","連続殺人","誘拐","盗難","毒殺"]
STYLES   = ["アガサクリスティ風","横溝正史風","北欧ミステリ風","倒叙形式","本格パズラー"]
TIMES    = ["冬の夜","秋の雨夜","夏の夕暮れ","嵐の夜","早朝"]
PLACES   = ["音楽ホール","山荘","列車個室","劇場楽屋","図書館","温泉旅館","高層マンション"]
CLUE_TYPES = ["key","footstep","log","weapon","fingerprint","witness","camera","document","audio","chemical"]

def parse_args():
    #パーサを作成
    p = argparse.ArgumentParser(description="事件仕様を手動指定（未指定はデフォルト）")
    #choicesはリスト内に制限,helpは説明文,--は入力順番関係なし
    p.add_argument("--genre",  choices=GENRES,  default="密室殺人",       help="事件ジャンル")
    p.add_argument("--style",  choices=STYLES,  default="アガサクリスティ風", help="作風")
    p.add_argument("--time",   choices=TIMES,   default="冬の夜",         help="舞台の時間")
    p.add_argument("--place",  choices=PLACES,  default="音楽ホール",     help="舞台の場所")
    
	#デフォルトでは文字列型なので整数値に指定
    p.add_argument("--suspects", type=int, default=3, help="容疑者数 S1..Sn")
    p.add_argument("--clues",    type=int, default=3, help="証拠数 C1..Cm")
    p.add_argument("--clue-types", default="key,log,footstep",help=f"証拠タイプをカンマで区切って入力。候補: {','.join(CLUE_TYPES)}")

    # importsのまま
    p.add_argument("--show-judge", action="store_true", help="Judgeの評価HUDを逐次表示")
    p.add_argument("--log-judge", default="", help="Judge評価をJSONLで保存するパス（例: runs/judge_log.jsonl）")

    return p.parse_args()#引数を解析