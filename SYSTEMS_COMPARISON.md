# 🏛️ 国交省データ RAGチャットボット - システム比較

このリポジトリには、**LangChain版**と**LlamaIndex版**の2つの独立したRAGシステムが含まれています。

## 📁 ディレクトリ構造

```
test_lang/
├── terminal_rag_chatbot.py      # LangChain版メインファイル
├── simple_rag_chatbot.py        # LangChain版Streamlit版
├── rag_chatbot.py              # LangChain版（詳細実装）
├── requirements.txt            # LangChain版依存関係
├── .env                       # LangChain版環境変数
└── llamaindex_system/         # 🦙 LlamaIndex版システム
    ├── llamaindex_rag_chatbot.py
    ├── requirements.txt
    ├── .env
    └── README.md
```

## 🔍 システム比較

| 特徴 | LangChain版 | LlamaIndex版 |
|------|-------------|--------------|
| **フレームワーク** | LangChain + 個別ライブラリ | LlamaIndex統合環境 |
| **PDF処理** | PyPDF2 + カスタム分割 | PyMuPDFReader + SentenceSplitter |
| **ベクトル検索** | FAISS + SentenceTransformers | VectorStoreIndex (内蔵) |
| **LLM統合** | OpenAI直接呼び出し | LlamaIndex LLMラッパー |
| **回答生成** | カスタムプロンプト | tree_summarizeモード |
| **ソース追跡** | 基本的な文書情報 | 詳細なメタデータ+スコア |
| **設定の柔軟性** | 高い（全て手動設定） | 中程度（統合された設定） |
| **学習コスト** | 高い（各コンポーネント理解） | 中程度（統一API） |

## 🚀 実行方法

### LangChain版の実行

```bash
# ルートディレクトリで
pip install -r requirements.txt
python terminal_rag_chatbot.py
```

### LlamaIndex版の実行

```bash
# llamaindex_systemディレクトリで
cd llamaindex_system
pip install -r requirements.txt
python llamaindex_rag_chatbot.py
```

## 💡 使い分けの指針

### LangChain版を選ぶべき場合
- **細かい制御が必要**: 各ステップを詳細にカスタマイズしたい
- **学習目的**: RAGの仕組みを深く理解したい
- **軽量化重視**: 必要最小限のライブラリで動作させたい
- **既存システム統合**: 他のLangChainベースシステムと統合

### LlamaIndex版を選ぶべき場合
- **迅速な開発**: 統合された環境で素早く構築したい
- **高度な機能**: tree_summarizeなどの高度な回答生成を使いたい
- **メタデータ重視**: 詳細なソース追跡が必要
- **スケーラビリティ**: 大規模な文書コレクションを扱う予定

## 🔧 環境設定

両システムで共通して必要：

1. **OpenAI APIキー**:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **PDFファイル**:
   ```
   /Users/yoshinomukanou/Downloads/国交省に関するデータ/国交省①.pdf
   ```

## 🎯 パフォーマンス特徴

### LangChain版
- ✅ 起動が早い
- ✅ メモリ使用量が少ない
- ✅ カスタマイズ性が高い
- ❌ 手動設定が多い

### LlamaIndex版
- ✅ 高度な回答品質
- ✅ 豊富な機能
- ✅ 統一されたAPI
- ❌ 起動時間が長い
- ❌ メモリ使用量が多い

## 🧪 どちらを試すべきか？

1. **RAG初心者**: まずLangChain版で仕組みを理解
2. **プロダクション**: 要件に応じて選択
3. **実験・研究**: 両方を試して比較検討

両システムは独立しているため、同じリポジトリ内で自由に切り替えて使用できます。