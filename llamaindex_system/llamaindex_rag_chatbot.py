#!/usr/bin/env python3
"""
国交省データ RAGチャットボット - LlamaIndex版
LlamaIndexを使用したRAGチャットボット
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
import logging
import sys

# Hugging Face Tokenizers警告を抑制
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ログレベル設定
logging.basicConfig(level=logging.WARNING)

load_dotenv()

class LlamaIndexRAGChatbot:
    def __init__(self):
        self.index = None
        self.query_engine = None
        
    def initialize(self):
        """LlamaIndexの設定を初期化"""
        print("🚀 LlamaIndexシステムを初期化しています...")
        
        try:
            # OpenAI API Key確認
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OpenAI APIキーが設定されていません")
                return False
            
            # LLM設定
            llm = OpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # エンベディング設定（多言語対応）
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # グローバル設定
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            print("✅ LlamaIndex初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {str(e)}")
            return False
    
    def load_and_index_pdf(self, pdf_path):
        """PDFを読み込んでインデックスを作成"""
        print(f"📄 PDFを読み込み中: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print("❌ PDFファイルが見つかりません")
            return False
        
        try:
            # PDFリーダー作成
            reader = PyMuPDFReader()
            
            # PDFを読み込み
            documents = reader.load_data(file_path=pdf_path)
            print(f"✅ {len(documents)}ページを読み込みました")
            
            # ノードパーサー設定
            node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            # インデックス作成
            print("🔄 ベクトルインデックスを作成中...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                node_parser=node_parser,
                show_progress=True
            )
            
            # クエリエンジン作成
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize"
            )
            
            print("✅ インデックス作成完了")
            return True
            
        except Exception as e:
            print(f"❌ PDF処理エラー: {str(e)}")
            return False
    
    def query(self, question):
        """質問に対する回答を生成"""
        if not self.query_engine:
            return "❌ システムが初期化されていません"
        
        try:
            # カスタムプロンプトで日本語回答を指定
            custom_prompt = f"""
以下の文書情報に基づいて、質問に日本語で詳しく回答してください。
文書に関連する情報がない場合は、「提供された文書にはその情報が含まれておりません」と回答してください。

質問: {question}

回答:"""
            
            response = self.query_engine.query(custom_prompt)
            return str(response)
            
        except Exception as e:
            return f"❌ 回答生成エラー: {str(e)}"
    
    def get_source_info(self, question):
        """ソース情報付きで回答を取得"""
        if not self.query_engine:
            return "❌ システムが初期化されていません", []
        
        try:
            response = self.query_engine.query(question)
            
            # ソース情報を抽出
            source_nodes = getattr(response, 'source_nodes', [])
            sources = []
            
            for node in source_nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    metadata = node.node.metadata
                    text = node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
                    
                    sources.append({
                        'page': metadata.get('page_label', '不明'),
                        'score': getattr(node, 'score', 0.0),
                        'text': text
                    })
            
            return str(response), sources
            
        except Exception as e:
            return f"❌ 回答生成エラー: {str(e)}", []
    
    def chat(self):
        """チャットループ"""
        print("\n" + "="*60)
        print("🦙 国交省データ RAGチャットボット (LlamaIndex版)")
        print("="*60)
        print("質問を入力してください。終了するには 'quit' または 'exit' を入力。")
        print("詳細情報を表示するには 'verbose' モードをオンにできます。")
        print("-"*60)
        
        verbose = False
        
        while True:
            try:
                user_input = input("\n💬 質問: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 チャットボットを終了します。")
                    break
                
                if user_input.lower() == 'verbose':
                    verbose = not verbose
                    status = "オン" if verbose else "オフ"
                    print(f"📚 詳細表示: {status}")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🔍 LlamaIndexで検索・回答生成中...")
                
                if verbose:
                    answer, sources = self.get_source_info(user_input)
                    
                    print(f"\n🦙 回答: {answer}")
                    
                    if sources:
                        print("\n📚 参照した文書:")
                        print("-" * 40)
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. ページ{source['page']} (スコア: {source['score']:.3f})")
                            print(f"   {source['text']}")
                            print()
                else:
                    answer = self.query(user_input)
                    print(f"\n🦙 回答: {answer}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 チャットボットを終了します。")
                break
            except Exception as e:
                print(f"❌ エラー: {str(e)}")

def main():
    print("🦙 国交省データ RAGチャットボット - LlamaIndex版")
    print("=" * 50)
    
    # OpenAI APIキー確認
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI APIキーが設定されていません。")
        print("💡 .envファイルに OPENAI_API_KEY=your_api_key を設定してください。")
        return
    
    # チャットボット初期化
    chatbot = LlamaIndexRAGChatbot()
    
    if not chatbot.initialize():
        return
    
    # PDF読み込み
    pdf_path = "/Users/yoshinomukanou/Downloads/国交省に関するデータ/国交省①.pdf"
    if not chatbot.load_and_index_pdf(pdf_path):
        return
    
    # チャット開始
    chatbot.chat()

if __name__ == "__main__":
    main()