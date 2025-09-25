# LLMデータセット生成ツール

これは、OpenAI互換の大規模言語モデル（LLM）を使用してデータセットを生成するための、柔軟なコマンドラインツールです。Hugging Faceのデータセットを入力データソースとして活用し、カスタマイズ可能なJinja2プロンプトテンプレートを介して処理を行い、LLMからの応答を生成して、さまざまなタスクのための新しいデータセットを作成します。

## 主な機能

*   **OpenAI互換APIへの接続**: ベースURLとAPIキーを設定することで、さまざまなLLMプロバイダーを利用できます。
*   **Hugging Faceデータセットの活用**: Hugging Face Hubの任意のデータセットをプロンプト生成のソースとして使用できます。
*   **カスタマイズ可能なプロンプト**: Jinja2テンプレートを使用して、ソースデータセットのデータをLLM用のプロンプトに簡単にフォーマットできます。
*   **テキストのチャンク化と分割**: データセットのカラムからテキストを段落、文、または行単位で自動的に小さなチャンクに分割し、より詳細なデータ生成を可能にします。
*   **堅牢性と効率性**:
    *   API呼び出しには指数関数的バックオフ付きの自動リトライ機能が含まれており、レート制限や一時的なエラーに対応します。
    *   データセットをストリーミングすることで、メモリ使用量を最小限に抑えます。
    *   長時間の生成実行中にデータが失われるのを防ぐため、定期的に進捗を保存します。
*   **使いやすさ**: シンプルなコマンドラインインターフェース。

## 前提条件

*   Python 3.8以上
*   pip

## インストール

1.  **リポジトリをクローンします:**
    ```bash
    git clone https://github.com/TylorShine/llm-dataset-generator.git
    cd llm-dataset-generator
    ```

2.  **仮想環境を作成して有効化します（推奨）:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate` を使用
    ```

3.  **必要な依存関係をインストールします:**
    ```bash
    pip install -r requirements.txt
    ```

## 設定

1.  **`.env`ファイルを作成します**。サンプルファイルをコピーしてください:
    ```bash
    cp .env.example .env
    ```

2.  **`.env`ファイルを編集して**、APIキーとLLMプロバイダーのベースURLを追加します:
    ```
    OPENAI_API_KEY="sk-your_api_key_here"
    OPENAI_BASE_URL="https://api.openai.com/v1"
    ```
    別のサービスを使用している場合は、`OPENAI_BASE_URL`を適切なエンドポイントに変更してください。

## 使用方法

メインスクリプトは`generate.py`です。コマンドラインからさまざまな引数を指定して実行し、データセット生成プロセスを制御できます。

### 基本的な例：質疑応答

この例では、`squad`データセットと`qa_template.yaml`を使用して、質問に対する回答を生成します。

```bash
python generate.py \
    --prompt-file prompts/qa_template.yaml \
    --output-file output/squad_qa_dataset.json \
    --dataset-name squad \
    --num-samples 50 \
    --model gpt-4o
```

### コマンドライン引数

*   `--prompt-file`: （必須）YAMLプロンプトテンプレートファイルへのパス。
*   `--output-file`: （必須）生成されたJSONデータセットを保存するパス。
*   `--dataset-name`: 使用するHugging Faceデータセットの名前（デフォルト: `squad`）。
*   `--dataset-split`: 使用するデータセットの分割（デフォルト: `train`）。
*   `--num-samples`: 生成するサンプル数（デフォルト: 10）。
*   `--start-index`: データセットの開始インデックス（デフォルト: 0）。
*   `--chunk-column`: 小さなチャンクに分割するカラムの名前。
*   `--chunk-by`: チャンク化の単位: `sentences`, `words`, `paragraphs`, `lines`（デフォルト: `lines`）。
*   `--chunk-size-min`/`--chunk-size-max`: チャンクごとの単位のランダムな範囲。
*   `--model`: 使用するモデルの名前（デフォルト: `gpt-3.5-turbo`）。
*   `--max-tokens`: 生成する最大トークン数（デフォルト: 512）。
*   `--temperature`: 生成時のサンプリング温度（デフォルト: 0.7）。
*   `--save-interval`: 中間データセットを保存する頻度（デフォルト: 10）。

## プロンプトテンプレートの作成方法

プロンプトテンプレートは、LLMのシステムプロンプトとユーザープロンプトを定義するYAMLファイルです。

1.  `prompts/`ディレクトリに新しい`.yaml`ファイルを作成します。
2.  `system_prompt`（任意）と`user_prompt_template`（必須）を定義します。
3.  Jinja2構文（`{{ column_name }}`）を使用して、ソースデータセットのデータを`user_prompt_template`に挿入します。

例として[`prompts/qa_template.yaml`](./prompts/qa_template.yaml)を参照してください。