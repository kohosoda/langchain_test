# LangChain プロジェクト

LangChainを使用したAIアプリケーション開発プロジェクト

## uvを使った環境構築

### 1. 初期セットアップ
```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 2. 基本的なコマンド

```bash
# 仮想環境でコマンド実行
uv run python main.py

# 新しいパッケージの追加
uv add package-name

# 開発用パッケージの追加
uv add --dev package-name

# パッケージの削除
uv remove package-name

# 依存関係の更新
uv lock --upgrade

# シェルの起動（仮想環境有効化）
uv shell
```

### 3. 環境変数設定

`.env`ファイルを作成してAPIキーを設定：
```
OPENAI_API_KEY=your_openai_api_key
```

#### LangGraphベースのRAGワークフロー

**重要**: LangGraphのRAGワークフローは相対インポートを使用しているため、**モジュールとして実行**する必要があります。

```bash
# srcディレクトリに移動してからモジュールとして実行
cd src
uv run python -m langgraph_rag.langgraph_rag
```

## プロジェクト構成

- `src/langgraph_rag/` - LangGraphベースのRAGワークフロー
  - `langgraph_rag.py` - メインワークフロー
  - `nodes/` - ワークフローノード
  - `routers/` - 判定ルーター
  - `models/` - データモデル
- `pyproject.toml` - プロジェクト設定とパッケージ管理
- `uv.lock` - 依存関係のロックファイル
