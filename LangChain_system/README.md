# 国交省データ RAGチャットボット - LangChain風実装

シンプルなターミナル版RAGチャットボットです。

## 🚀 実行

```bash
pip install -r requirements.txt
python terminal_rag_chatbot.py
```

## 🔧 環境設定

`.env`ファイルにAPIキーを設定：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 💡 機能

- 📄 PDF自動処理
- 🔍 ベクトル検索（FAISS）
- 💬 日本語質問応答
- 📚 参照文書表示（verboseモード）

## ⚠️ 注意

- OpenAI APIキーが必要
- 初回起動は時間がかかる場合があります