# 国交省データ RAGチャットボット - LlamaIndex版

LlamaIndexを使用した国土交通省のPDFデータに基づく質問応答システムです。

## 🦙 LlamaIndexの特徴

- **統合された文書処理**: PDFリーダー、チャンクング、ベクトル化が一体化
- **高度なクエリエンジン**: tree_summarizeモードで詳細な回答生成
- **柔軟な設定**: チャンクサイズ、重複、類似度などを詳細に調整可能
- **リッチなメタデータ**: ページ情報とスコアを含む詳細なソース情報

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
cd llamaindex_system
pip install -r requirements.txt
```

### 2. 環境設定

`.env` ファイルにOpenAI APIキーを設定：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 実行

```bash
python llamaindex_rag_chatbot.py
```

## 💡 機能

- 📄 **高度なPDF処理**: PyMuPDFReaderによる精密な文書読み込み
- 🧠 **インテリジェントチャンクング**: SentenceSplitterによる意味的分割
- 🔍 **ベクトル検索**: similarity_top_k=3での関連文書検索
- 💬 **Tree Summarization**: 複数文書からの包括的な回答生成
- 📊 **詳細ソース情報**: ページ番号とスコア付きの参照情報

## 🔧 技術仕様

- **フレームワーク**: LlamaIndex 0.10.0
- **PDF処理**: PyMuPDFReader
- **チャンクング**: SentenceSplitter (512文字, 重複50文字)
- **エンベディング**: HuggingFace多言語MiniLM
- **LLM**: OpenAI GPT-3.5-turbo
- **クエリエンジン**: VectorStoreIndex + tree_summarize

## 📖 使用方法

1. システムが自動的に初期化されます
2. PDFが読み込まれ、ベクトルインデックスが作成されます
3. 質問を入力すると、関連文書から回答が生成されます
4. `verbose` コマンドで詳細なソース情報を表示できます
5. `quit` で終了

## ⚠️ 注意事項

- 初回起動時はモデルダウンロードのため時間がかかります
- LlamaIndexは比較的大きなライブラリです
- tree_summarizeモードのため、複数の関連文書から包括的な回答を生成します