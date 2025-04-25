import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. APIキーの設定
# 安全のため、環境変数から読み込むことを推奨します。
# 環境変数 'OPENROUTER_API_KEY' にキーを設定してください。
# もし直接コードに書く場合は、 api_key="YOUR_OPENROUTER_API_KEY" のように指定しますが、
# 公開リポジトリなどには絶対にコミットしないでください。
# .envファイルから環境変数を読み込む
# このスクリプトと同じディレクトリか親ディレクトリにある .env ファイルを探します
load_dotenv()

# 1. APIキーの設定 (環境変数から読み込み)
# load_dotenv() によって .env ファイルの内容が環境変数として読み込まれている
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("APIキーが設定されていません。環境変数 'OPENROUTER_API_KEY' を設定してください。")

# 2. OpenAIクライアントの初期化 (OpenRouter用に設定)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", # OpenRouterのエンドポイント
    api_key=api_key,
)


prompt = """I want to find a formula that evaluates to 24 using the following four numbers and operations of "+", "-", "*", "/", "(", ")", and "=".
Note that we must use each number exactly once to build the formula.

Numbers: [3, 9, 8, 8]

I tried building a formula: 3*9-8/8

Please check if my formula is correct or not.
Then please give me helpful hints or direction on how to find the formula effectively if my formula is incorrect.
Note that I want to improve my thought myself so please do not include answers in your feedback directly.
"""

prompt2 = """I want to find a formula that evaluates to 24 using the following four numbers and operations of "+", "-", "*", "/", "(", ")", and "=".
Note that we must use each number exactly once to build the formula.

Numbers: [5, 9, 8, 8]

I tried building a formula: 5*9-8/8

Please check if my formula is correct or not.
Then please give me helpful hints or direction on how to find the formula effectively if my formula is incorrect.
Note that I want to improve my thought myself so please do not include answers in your feedback directly.
"""

# 3. API呼び出し (Chat Completion)
try:
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",  # 使用したいモデルのIDを指定
        messages=[
            # {"role": "system", "content": "あなたは役立つアシスタントです。"}, # 必要に応じてシステムプロンプトを設定
            {"role": "user", "content": prompt},
            {"role": "user", "content": prompt2},
        ],
        # --- オプション: その他のパラメータ ---
        # temperature=0.7,  # 生成のランダム性 (0に近いほど決定的)
        # max_tokens=1000,   # 最大生成トークン数
        # stream=False,     # ストリーミング応答にするか (True/False)
        # ---------------------------------
    )

    # 4. 応答の表示
    # print(response) # 完全なレスポンスオブジェクトを確認したい場合
    if response.choices:
        print("AIの応答:")
        print(response.choices[0].message.content)
    else:
        print("応答がありませんでした。")

    import pdb; pdb.set_trace()

    # --- オプション: 使用状況の表示 ---
    # usage = response.usage
    # if usage:
    #     print("\n--- Usage ---")
    #     print(f"Prompt Tokens: {usage.prompt_tokens}")
    #     print(f"Completion Tokens: {usage.completion_tokens}")
    #     print(f"Total Tokens: {usage.total_tokens}")
    # -------------------------------

except Exception as e:
    print(f"エラーが発生しました: {e}")