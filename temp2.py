import json

def transform_json_format(input_data):
    """
    指定された古い形式のJSONデータを新しい形式に変換します。

    Args:
        input_data (list): 古い形式の辞書を含むリスト。
                           各辞書は 'instruction', 'answer', 'image_path' キーを持つ想定。

    Returns:
        list: 新しい形式の辞書を含むリスト。
              各辞書は 'image' と 'conversations' キーを持つ。
    """
    output_data = []
    for item in input_data:
        # 安全のため、キーが存在しない場合に備えて .get() を使用します
        instruction = item.get('instruction', '')
        answer = item.get('answer', '')
        image_path = item.get('image_path', None) # image_pathは必須かもしれないのでNoneをデフォルトに

        if image_path is None:
            print(f"警告: 'image_path' が見つからないアイテムがあります: {item}")
            continue # image_pathがない場合はスキップするか、エラー処理を追加

        # 新しい形式の辞書を作成
        new_item = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    # instructionの先頭に "<image>\n" を追加
                    "value": f"<image>\n{instruction}"
                },
                {
                    "from": "gpt",
                    "value": answer.replace("\"}", "=24\"}")
                    # "value": answer
                }
            ]
        }
        output_data.append(new_item)
    return output_data


# # JSON文字列をPythonのリストに変換
# try:
#     original_data = json.loads(input_json_str)
# except json.JSONDecodeError as e:
#     print(f"JSONデータの読み込み中にエラーが発生しました: {e}")
#     original_data = [] # エラー時は空リストで続行するか、処理を中断

# if original_data:
#     # データ形式を変換
#     transformed_data = transform_json_format(original_data)

#     # 変換後のデータをJSON形式の文字列として出力（見やすいようにインデント付き）
#     # ensure_ascii=False で日本語などが正しく表示されるようにします
#     output_json_str = json.dumps(transformed_data, indent=4, ensure_ascii=False)
#     print(output_json_str)

# --- ファイルからの読み込みと書き込みを行う場合 ---
def transform_json_file(input_filepath, output_filepath):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            original_data = json.load(infile)

        transformed_data = transform_json_format(original_data)
        print(f"変換前のデータ数: {len(original_data)}")
        print(f"変換後のデータ数: {len(transformed_data)}")

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)

        print(f"変換完了: {input_filepath} -> {output_filepath}")

    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません - {input_filepath}")
    except json.JSONDecodeError as e:
        print(f"JSONデータの読み込み中にエラーが発生しました ({input_filepath}): {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")




# # 使用例：
input_file = '/home/suganuma/datasets/card_24/sft/train_v2/sft_data.json'   # 元のJSONファイル名
output_file = '/home/suganuma/datasets/card_24/sft/train_v2/sft_data_conv.json' # 出力するJSONファイル名
transform_json_file(input_file, output_file) # この行を実行するには、input.jsonファイルが必要です。